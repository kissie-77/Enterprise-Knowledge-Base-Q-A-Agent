"""
RAG 问答模块
"""
import re
from typing import Dict, List, Any

from upload import get_embedding_model, get_chroma_client, COLLECTION_NAME
from llm_client import call_zhipu_llm
from tools import get_current_time, calculate


def retrieve_similar_chunks(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    model = get_embedding_model()
    question_embedding = model.encode([question])[0].tolist()

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            chunks.append({
                "content": doc,
                "score": results['distances'][0][i] if results.get('distances') else 0
            })

    return chunks


def needs_tool_call(question: str) -> tuple[bool, str]:
    question_lower = question.lower()

    time_patterns = [
        r'现在几点', r'当前时间', r'几点了', r'现在几点了',
        r'what time', r'current time', r'time now'
    ]
    for pattern in time_patterns:
        if re.search(pattern, question_lower):
            return True, "get_current_time"

    calc_patterns = [
        r'计算', r'等于多少', r'等于', r'结果是', r'是多少',
        r'calculate', r'compute', r'what is'
    ]
    for pattern in calc_patterns:
        if re.search(pattern, question_lower):
            if re.search(r'\d', question) and re.search(r'[+\-*/]', question):
                return True, "calculate"

    return False, ""


def execute_tool(tool_name: str, question: str) -> str:
    if tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "calculate":
        match = re.search(r'([\d\.\+\-\*/\(\)]+)', question)
        if match:
            expression = match.group(1)
            try:
                return calculate(expression)
            except Exception as e:
                return f"计算错误: {str(e)}"
        else:
            return "无法从问题中提取计算表达式"
    return f"未知工具: {tool_name}"


SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请严格基于参考资料来回答用户的问题。

回答要求：
1. 详细、完整地回答，不要遗漏重要信息
2. 充分引用和展开说明参考资料中的内容
3. 如果有多个相关片段，综合整理后给出全面回答
4. 使用清晰的结构（分点、分段）组织回答
5. 如果参考资料中完全没有相关信息，请明确说明"抱歉，知识库中暂无相关信息"

请用中文回答。"""


def build_prompt(context: str, question: str, tool_result: str = None) -> str:
    if tool_result:
        return f"""你是一个专业的智能助手。以下是工具调用的结果：

工具结果：
{tool_result}

用户问题：{question}

请基于工具结果，给出详细、准确的回答。"""
    else:
        return f"""{SYSTEM_PROMPT}

参考资料：
{context}

用户问题：{question}"""


async def answer_question(question: str) -> Dict[str, Any]:
    result = {
        "answer": "",
        "contexts": [],
        "tool_used": None
    }

    needs_tool, tool_name = needs_tool_call(question)

    if needs_tool:
        tool_result = execute_tool(tool_name, question)
        result["tool_used"] = tool_name
        prompt = build_prompt("", question, tool_result)
        answer = await call_zhipu_llm(prompt)
        result["answer"] = answer
    else:
        chunks = retrieve_similar_chunks(question, top_k=6)

        if not chunks:
            prompt = build_prompt("没有找到相关文档内容", question)
        else:
            context = "\n\n".join([f"参考资料{i+1}：{chunk['content']}" for i, chunk in enumerate(chunks)])
            result["contexts"] = [{"content": chunk["content"], "score": chunk["score"]} for chunk in chunks]
            prompt = build_prompt(context, question)

        answer = await call_zhipu_llm(prompt)
        result["answer"] = answer

    return result
