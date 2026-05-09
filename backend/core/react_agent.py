"""
ReAct 智能体模块
实现 Reasoning + Acting 范式，参考 hello-agents 的 ReActAgent 设计

ReAct 循环:
1. Thought（思考）: 分析问题，规划下一步
2. Action（行动）: 调用工具获取信息
3. Observation（观察）: 获取工具返回结果
4. 重复直到得出最终答案 → Finish[答案]
"""
import re
from typing import Optional, List, Dict, Any

from .base_agent import BaseAgent, Message
from .llm_client import LLMClient
from .tool_registry import ToolRegistry


# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """你是一个具备推理和行动能力的智能问答助手。你可以通过思考来分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 回答格式
请严格按照以下格式进行回应，每次只执行一个步骤：

Thought: 你的思考过程，分析问题并规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `工具名[参数]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时使用

## 重要规则
1. 每次回应必须包含 Thought 和 Action 两部分
2. 工具调用格式必须严格遵循: 工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用 Finish[最终答案]
4. 如果工具返回信息不够，继续思考并使用其他工具
5. 最终答案应当详细、准确、有条理

## 当前问题
Question: {question}

## 已有的推理历史
{history}

现在开始你的推理和行动："""


# 带 RAG 上下文的 ReAct 提示词
REACT_RAG_PROMPT_TEMPLATE = """你是一个企业知识库智能问答助手。你可以结合知识库中检索到的内容和工具来回答用户问题。

## 知识库检索结果
{context}

## 可用工具
{tools}

## 回答格式
请严格按照以下格式进行回应：

Thought: 分析用户问题，判断知识库内容是否足够回答。如果足够，直接给出最终答案；如果不够，使用工具补充信息。
Action: 你的行动，格式如下：
- `工具名[参数]` - 调用工具
- `Finish[最终答案]` - 给出最终答案

## 重要规则
1. 优先使用知识库中的信息回答
2. 如果知识库信息不足，再使用工具补充
3. 回答要详细、准确、有条理
4. 引用知识库内容时要忠于原文

## 当前问题
Question: {question}

## 推理历史
{history}

现在开始你的推理和行动："""


class ReActAgent(BaseAgent):
    """
    ReAct 智能体
    结合推理（Reasoning）和行动（Acting）来解决复杂问题

    特点：
    - 支持多步推理
    - 可调用注册的工具
    - 可结合 RAG 检索结果
    - 保留思考链用于可解释性
    """

    def __init__(
        self,
        name: str = "ReActAgent",
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
        verbose: bool = True,
    ):
        """
        初始化 ReAct 智能体

        Args:
            name: 智能体名称
            llm: LLM 客户端
            tool_registry: 工具注册表
            system_prompt: 系统提示（可选）
            max_steps: 最大推理步数
            verbose: 是否打印详细推理过程
        """
        super().__init__(name=name, system_prompt=system_prompt)
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.verbose = verbose

        # 记录推理过程
        self._reasoning_trace: List[Dict[str, str]] = []

    async def run(self, input_text: str, context: str = "", **kwargs) -> str:
        """
        运行 ReAct 推理循环

        Args:
            input_text: 用户问题
            context: RAG 检索到的上下文（可选）
            **kwargs: 额外参数

        Returns:
            最终答案
        """
        self._reasoning_trace = []
        history_lines: List[str] = []
        current_step = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 {self.name} 开始处理问题: {input_text}")
            print(f"{'='*60}")

        while current_step < self.max_steps:
            current_step += 1

            if self.verbose:
                print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            prompt = self._build_prompt(input_text, history_lines, context)

            # 2. 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = await self.llm.chat(messages, temperature=0)

            if not response_text:
                if self.verbose:
                    print("❌ LLM 未返回有效响应")
                break

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            if self.verbose:
                if thought:
                    print(f"🤔 思考: {thought}")

            # 记录推理过程
            self._reasoning_trace.append({
                "step": current_step,
                "thought": thought or "",
                "action": action or "",
                "raw_response": response_text,
            })

            if not action:
                if self.verbose:
                    print("⚠️ 未能解析出有效的 Action")
                # 尝试将整个回复作为最终答案
                final_answer = thought or response_text
                self._save_to_history(input_text, final_answer)
                return final_answer

            # 4. 检查是否完成
            if action.startswith("Finish"):
                final_answer = self._extract_finish_content(action)
                if self.verbose:
                    print(f"🎉 最终答案: {final_answer[:200]}...")
                self._reasoning_trace[-1]["final_answer"] = final_answer
                self._save_to_history(input_text, final_answer)
                return final_answer

            # 5. 执行工具调用
            tool_name, tool_input = self._parse_action(action)

            if not tool_name:
                history_lines.append(f"Thought: {thought}")
                history_lines.append(f"Action: {action}")
                history_lines.append("Observation: 无效的 Action 格式，请使用正确格式: 工具名[参数]")
                continue

            if self.verbose:
                print(f"🔧 调用工具: {tool_name}[{tool_input}]")

            # 执行工具
            observation = await self.tool_registry.execute_tool(tool_name, tool_input)

            if self.verbose:
                print(f"👀 观察结果: {observation[:200]}")

            # 更新历史
            history_lines.append(f"Thought: {thought}")
            history_lines.append(f"Action: {tool_name}[{tool_input}]")
            history_lines.append(f"Observation: {observation}")

            self._reasoning_trace[-1]["observation"] = observation

        # 达到最大步数，生成最终回答
        if self.verbose:
            print(f"\n⚠️ 已达到最大步数 ({self.max_steps})，尝试生成最终回答...")

        final_answer = await self._force_final_answer(input_text, history_lines, context)
        self._save_to_history(input_text, final_answer)
        return final_answer

    def _build_prompt(self, question: str, history: List[str], context: str = "") -> str:
        """构建提示词"""
        tools_desc = self.tool_registry.get_tools_description()
        history_str = "\n".join(history) if history else "（暂无历史）"

        if context:
            return REACT_RAG_PROMPT_TEMPLATE.format(
                context=context,
                tools=tools_desc,
                question=question,
                history=history_str,
            )
        else:
            return REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
            )

    async def _force_final_answer(
        self, question: str, history: List[str], context: str = ""
    ) -> str:
        """强制生成最终回答（达到最大步数时使用）"""
        history_str = "\n".join(history)
        
        prompt = f"""基于以下推理过程，请直接给出最终答案。

问题: {question}

推理历史:
{history_str}

{"知识库参考:\n" + context if context else ""}

请直接给出完整、准确的最终答案（不需要再使用 Thought/Action 格式）："""

        messages = [{"role": "user", "content": prompt}]
        return await self.llm.chat(messages, temperature=0)

    def _parse_output(self, text: str):
        """解析 LLM 输出中的 Thought 和 Action"""
        # 匹配 Thought
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL
        )
        # 匹配 Action
        action_match = re.search(
            r"Action:\s*(.*?)$", text, re.DOTALL
        )

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    def _parse_action(self, action_text: str):
        """解析 Action 中的工具名和参数"""
        # 匹配: 工具名[参数]
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _extract_finish_content(self, action_text: str) -> str:
        """提取 Finish[...] 中的内容"""
        match = re.match(r"Finish\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1)
        # 如果格式不完全匹配，尝试提取方括号后的内容
        match = re.match(r"Finish\s*\[?(.*)", action_text, re.DOTALL)
        if match:
            content = match.group(1).rstrip("]")
            return content
        return action_text.replace("Finish", "").strip("[] ")

    def _save_to_history(self, question: str, answer: str):
        """保存问答对到历史记录"""
        self.add_message(Message(content=question, role="user"))
        self.add_message(Message(content=answer, role="assistant"))

    def get_reasoning_trace(self) -> List[Dict[str, str]]:
        """获取最近一次的完整推理链"""
        return self._reasoning_trace.copy()

    def get_trace_summary(self) -> str:
        """获取推理过程摘要"""
        if not self._reasoning_trace:
            return "暂无推理记录"

        summary_parts = []
        for step in self._reasoning_trace:
            step_num = step["step"]
            thought = step.get("thought", "")[:100]
            action = step.get("action", "")[:80]
            obs = step.get("observation", "")[:80]

            part = f"步骤{step_num}: {thought}"
            if action:
                part += f"\n  行动: {action}"
            if obs:
                part += f"\n  观察: {obs}"
            summary_parts.append(part)

        return "\n".join(summary_parts)
