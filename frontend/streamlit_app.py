"""
企业知识库问答 Agent - Streamlit 前端
基于 hello-agents 架构的交互式界面

功能：
- 智能问答（ReAct Agent + RAG）
- 文档上传与知识库管理
- 推理过程展示（思考链可视化）
- 多轮对话支持
- 评测面板
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import uuid

# 页面配置
st.set_page_config(
    page_title="企业知识库问答 Agent",
    page_icon="🤖",
    layout="wide",
)

# 后端 API 地址
# 部署时通过 Nginx 反向代理，前端访问 /api/ 路径即可
# 本地开发时可改为 http://localhost:8000
API_BASE_URL = st.sidebar.text_input(
    "API 地址",
    value="https://agent.wenhuichen.cn/api",
    help="后端服务地址（部署环境使用 https://agent.wenhuichen.cn/api）",
)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]


def sidebar_config():
    """侧边栏配置"""
    st.sidebar.title("🤖 智能问答助手")
    st.sidebar.caption("基于 hello-agents 架构")

    st.sidebar.markdown("---")

    # Agent 设置
    st.sidebar.markdown("##### ⚙️ Agent 设置")
    use_rag = st.sidebar.checkbox("启用知识库检索", value=True, help="从上传的文档中检索相关内容")
    use_agent = st.sidebar.checkbox("启用 ReAct 推理", value=True, help="使用多步推理和工具调用")
    top_k = st.sidebar.slider("检索数量", 1, 10, 5, help="返回最相关的文档片段数")

    st.session_state.use_rag = use_rag
    st.session_state.use_agent = use_agent
    st.session_state.top_k = top_k

    # 会话管理
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 💬 会话管理")
    st.sidebar.text(f"会话 ID: {st.session_state.session_id}")
    if st.sidebar.button("🔄 新建会话"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        # 通知后端清除会话
        try:
            requests.delete(f"{API_BASE_URL}/sessions/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.rerun()

    # 工具信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🔧 可用工具")
    try:
        resp = requests.get(f"{API_BASE_URL}/tools", timeout=5)
        if resp.status_code == 200:
            tools = resp.json().get("tools", [])
            for tool in tools:
                st.sidebar.markdown(f"- **{tool['name']}**: {tool['description'][:40]}...")
        else:
            st.sidebar.info("无法获取工具列表")
    except Exception:
        st.sidebar.info("后端未连接")


def upload_handler():
    """文档上传"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📄 上传文档")
    uploaded_file = st.sidebar.file_uploader(
        "选择文件",
        type=["pdf", "txt", "docx", "md"],
        label_visibility="collapsed",
    )

    col1, col2 = st.sidebar.columns(2)
    chunk_size = col1.number_input("分块大小", 200, 2000, 500, step=100)
    overlap = col2.number_input("重叠字符", 0, 200, 50, step=10)

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_file_id") != file_id:
            if st.sidebar.button("📤 上传并处理", type="primary"):
                with st.spinner("正在处理文档..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        data = {"chunk_size": str(chunk_size), "overlap": str(overlap)}
                        response = requests.post(
                            f"{API_BASE_URL}/upload",
                            files=files,
                            data=data,
                            timeout=120,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.uploaded_file_id = file_id
                            st.sidebar.success(f"✅ {result['message']}")
                        else:
                            error_detail = response.json().get("detail", "未知错误")
                            st.sidebar.error(f"上传失败: {error_detail}")
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("无法连接后端服务，请检查 API 地址")
                    except Exception as e:
                        st.sidebar.error(f"错误: {str(e)}")
        else:
            st.sidebar.info(f"✅ 已上传: {uploaded_file.name}")


def chat_interface():
    """主对话界面"""
    st.title("🤖 企业知识库问答 Agent")
    st.caption("基于 ReAct + RAG + Memory 的智能问答系统 | hello-agents 架构")

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示推理过程
            if message.get("reasoning_trace"):
                with st.expander("🧠 推理过程"):
                    st.text(message["reasoning_trace"])

            # 显示检索到的上下文
            if message.get("contexts"):
                with st.expander(f"📚 参考资料 ({len(message['contexts'])} 条)"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        score = ctx.get("score", 0)
                        st.markdown(f"**片段 {i}** (相似度距离: {score:.4f})")
                        st.markdown(f"> {ctx['content'][:300]}...")
                        st.markdown("---")

            # 显示元信息
            if message.get("tool_used"):
                st.caption(f"🔧 使用工具: {message['tool_used']}")
            if message.get("response_time"):
                st.caption(f"⏱️ 响应时间: {message['response_time']}ms")

    # 用户输入
    prompt = st.chat_input("请输入您的问题...")

    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用后端
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "use_rag": st.session_state.get("use_rag", True),
                            "use_agent": st.session_state.get("use_agent", True),
                            "top_k": st.session_state.get("top_k", 5),
                        },
                        timeout=60,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        contexts = result.get("contexts", [])
                        tool_used = result.get("tool_used")
                        response_time = result.get("response_time_ms", 0)
                        reasoning_trace = result.get("reasoning_trace")

                        # 显示回答
                        st.markdown(answer)

                        # 推理过程
                        if reasoning_trace:
                            with st.expander("🧠 推理过程"):
                                st.text(reasoning_trace)

                        # 参考资料
                        if contexts:
                            with st.expander(f"📚 参考资料 ({len(contexts)} 条)"):
                                for i, ctx in enumerate(contexts, 1):
                                    score = ctx.get("score", 0)
                                    st.markdown(f"**片段 {i}** (距离: {score:.4f})")
                                    st.markdown(f"> {ctx['content'][:300]}...")
                                    st.markdown("---")

                        # 元信息
                        info_parts = []
                        if tool_used:
                            info_parts.append(f"🔧 {tool_used}")
                        info_parts.append(f"⏱️ {response_time}ms")
                        st.caption(" | ".join(info_parts))

                        # 保存到历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "contexts": contexts,
                            "tool_used": tool_used,
                            "response_time": response_time,
                            "reasoning_trace": reasoning_trace,
                        })

                        # 反馈按钮
                        col1, col2, _ = st.columns([1, 1, 6])
                        with col1:
                            if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                                submit_feedback(prompt, answer, "upvote", response_time, len(contexts))
                                st.toast("感谢您的反馈！")
                        with col2:
                            if st.button("👎", key=f"dn_{len(st.session_state.messages)}"):
                                submit_feedback(prompt, answer, "downvote", response_time, len(contexts))
                                st.toast("感谢您的反馈，我们会持续改进！")

                    else:
                        error_msg = response.json().get("detail", "未知错误")
                        st.error(f"请求失败: {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ 处理请求时出现错误: {error_msg}",
                        })

                except requests.exceptions.ConnectionError:
                    st.error("无法连接后端服务，请确保后端已启动")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ 无法连接后端服务",
                    })
                except requests.exceptions.Timeout:
                    st.error("请求超时，请稍后重试")
                except Exception as e:
                    st.error(f"错误: {str(e)}")


def submit_feedback(question, answer, feedback, response_time, retrieved_chunks):
    """提交反馈"""
    try:
        requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "question": question,
                "answer": answer,
                "feedback": feedback,
                "response_time_ms": response_time,
                "retrieved_chunks": retrieved_chunks,
            },
            timeout=10,
        )
    except Exception:
        pass


def evaluation_dashboard():
    """评测面板"""
    st.header("📊 评测面板")

    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            stats = data.get("stats", {})
            rag_stats = data.get("rag", {})

            # 核心指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总请求数", stats.get("total_requests", 0))
            with col2:
                st.metric("点赞率", f"{stats.get('upvote_rate', 0)}%")
            with col3:
                st.metric("平均响应时间", f"{stats.get('avg_response_time_ms', 0)}ms")
            with col4:
                st.metric("知识库片段数", rag_stats.get("total_chunks", 0))

            # 系统信息
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 系统信息")
                st.markdown(f"- **活跃会话数**: {data.get('active_sessions', 0)}")
                st.markdown(f"- **已注册工具**: {', '.join(data.get('tools', []))}")
                st.markdown(f"- **嵌入模型**: {rag_stats.get('embedding_model', 'N/A')}")
            with col2:
                st.markdown("##### 反馈统计")
                st.markdown(f"- **👍 点赞**: {stats.get('upvotes', 0)}")
                st.markdown(f"- **👎 点踩**: {stats.get('downvotes', 0)}")

            # 趋势图
            st.markdown("---")
            st.subheader("近7天趋势")
            if stats.get("daily_stats"):
                df = pd.DataFrame(stats["daily_stats"])
                fig = px.bar(
                    df, x="date", y="count",
                    title="每日请求数",
                    labels={"date": "日期", "count": "请求数"},
                )
                st.plotly_chart(fig, use_container_width=True)

                df_melted = df.melt(
                    id_vars=["date"],
                    value_vars=["upvotes", "downvotes"],
                    var_name="type",
                    value_name="count",
                )
                fig2 = px.bar(
                    df_melted, x="date", y="count", color="type",
                    title="每日反馈",
                    labels={"date": "日期", "count": "数量", "type": "类型"},
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("暂无数据，请先进行问答交互")

            # 最近反馈
            st.subheader("最近反馈记录")
            if stats.get("recent_feedback"):
                feedback_df = pd.DataFrame(stats["recent_feedback"])
                st.dataframe(
                    feedback_df[["id", "question", "feedback", "created_at"]],
                    use_container_width=True,
                )
            else:
                st.info("暂无反馈记录")
        else:
            st.error("无法获取统计数据")

    except requests.exceptions.ConnectionError:
        st.error("无法连接后端服务")
    except Exception as e:
        st.error(f"获取统计信息失败: {str(e)}")


def architecture_page():
    """架构说明页面"""
    st.header("🏗️ 系统架构")

    st.markdown("""
    ## 基于 hello-agents 的企业知识库问答系统

    本系统参考 [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) 
    教程架构，构建了一个完整的企业级知识库问答 Agent。

    ### 核心架构

    ```
    用户输入 → ReAct Agent（推理循环）
                    ├── 思考（Thought）: 分析问题
                    ├── 行动（Action）: 调用工具或检索知识库
                    ├── 观察（Observation）: 获取结果
                    └── 最终答案（Finish）: 输出回答
    ```

    ### 模块说明

    | 模块 | 说明 |
    |------|------|
    | **LLMClient** | 通用 LLM 客户端，支持智谱/DeepSeek/通义千问/Moonshot/OpenAI |
    | **ToolRegistry** | 工具注册表，支持装饰器/继承/函数三种注册方式 |
    | **ReActAgent** | ReAct 智能体，Reasoning + Acting 推理范式 |
    | **RAGEngine** | RAG 检索增强生成引擎，多格式文档 + 智能分块 |
    | **Memory** | 记忆模块，工作记忆/情景记忆/语义记忆三层体系 |

    ### 内置工具

    - 🕐 **get_current_time**: 获取当前时间
    - 🧮 **calculator**: 安全数学计算
    - 🔍 **web_search**: 网页搜索（需配置 SerpAPI）

    ### 扩展指南

    你可以轻松添加自定义工具：

    ```python
    from core.tool_registry import BaseTool

    class MyTool(BaseTool):
        name = "my_tool"
        description = "我的自定义工具"

        async def execute(self, input_text: str, **kwargs) -> str:
            # 你的工具逻辑
            return "结果"

    # 在 main.py 中注册
    tool_registry.register(MyTool())
    ```
    """)


def main():
    """主函数"""
    init_session_state()

    # 侧边栏
    sidebar_config()
    upload_handler()

    # 页面导航
    st.sidebar.markdown("---")
    page = st.sidebar.radio("📑 导航", ["💬 问答", "📊 评测面板", "🏗️ 架构说明"])

    # 路由
    if page == "💬 问答":
        chat_interface()
    elif page == "📊 评测面板":
        evaluation_dashboard()
    elif page == "🏗️ 架构说明":
        architecture_page()


if __name__ == "__main__":
    main()
