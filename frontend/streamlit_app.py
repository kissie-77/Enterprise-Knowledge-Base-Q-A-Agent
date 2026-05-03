"""
企业知识库问答 Agent - Streamlit 前端
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="企业知识库问答 Agent",
    page_icon="🤖",
    layout="wide"
)

# 后端 API 地址
API_BASE_URL = "https://agent.wenhuichen.cn/api"


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def upload_handler():
    """侧边栏上传处理"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 上传文档")
    uploaded_file = st.sidebar.file_uploader(
        "选择 PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        label_visibility="visible"
    )
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_file_id") != file_id:
            with st.spinner("正在处理文档..."):
                try:
                    files = {"file": uploaded_file}
                    data = {"chunk_size": "500", "overlap": "50"}
                    response = requests.post(
                        f"{API_BASE_URL}/upload",
                        files=files, data=data, timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.uploaded_file_id = file_id
                        st.sidebar.success(f"{result['message']}")
                    else:
                        error_detail = response.json().get('detail', '未知错误')
                        st.sidebar.error(f"上传失败: {error_detail}")
                except Exception as e:
                    st.sidebar.error(f"连接失败: {str(e)}")
        else:
            st.sidebar.info(f"已上传: {uploaded_file.name}")


def chat_interface():
    """聊天问答界面"""
    st.title("企业知识库问答 Agent")

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "contexts" in message and message["contexts"]:
                with st.expander("检索到的上下文"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        st.markdown(f"**片段 {i}:** {ctx['content']}")

    # chat_input 在顶层调用，自动固定在页面底部
    prompt = st.chat_input("请输入您的问题...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        contexts = result.get("contexts", [])
                        tool_used = result.get("tool_used")
                        response_time = result.get("response_time_ms", 0)

                        st.markdown(answer)

                        if tool_used:
                            st.info(f"使用了工具: {tool_used}")
                        st.caption(f"响应时间: {response_time}ms")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "contexts": contexts,
                            "tool_used": tool_used,
                            "response_time": response_time
                        })

                        # 反馈按钮
                        col1, col2, _ = st.columns([1, 1, 4])
                        with col1:
                            if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                                submit_feedback(prompt, answer, "upvote", response_time, len(contexts))
                        with col2:
                            if st.button("👎", key=f"dn_{len(st.session_state.messages)}"):
                                submit_feedback(prompt, answer, "downvote", response_time, len(contexts))

                    else:
                        error_msg = response.json().get("detail", "未知错误")
                        st.error(f"请求失败: {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"抱歉，处理请求时出现错误: {error_msg}"
                        })

                except Exception as e:
                    st.error(f"连接后端服务失败: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"连接后端服务失败: {str(e)}"
                    })


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
                "retrieved_chunks": retrieved_chunks
            },
            timeout=10
        )
    except Exception:
        pass


def evaluation_dashboard():
    """评测面板"""
    st.header("评测面板")

    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()["stats"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总请求数", stats["total_requests"])
            with col2:
                st.metric("点赞率", f"{stats['upvote_rate']}%")
            with col3:
                st.metric("平均响应时间", f"{stats['avg_response_time_ms']}ms")
            with col4:
                st.metric("点赞/点踩", f"{stats['upvotes']} / {stats['downvotes']}")

            st.subheader("近7天趋势")
            if stats["daily_stats"]:
                df = pd.DataFrame(stats["daily_stats"])
                fig = px.bar(df, x="date", y="count", title="每日请求数",
                             labels={"date": "日期", "count": "请求数"})
                st.plotly_chart(fig, use_container_width=True)

                df_melted = df.melt(id_vars=["date"], value_vars=["upvotes", "downvotes"],
                                    var_name="type", value_name="count")
                fig2 = px.bar(df_melted, x="date", y="count", color="type",
                              title="每日点赞/点踩数",
                              labels={"date": "日期", "count": "数量", "type": "类型"})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("暂无数据，请先进行问答")

            st.subheader("最近反馈记录")
            if stats["recent_feedback"]:
                feedback_df = pd.DataFrame(stats["recent_feedback"])
                st.dataframe(feedback_df[["id", "question", "feedback", "created_at"]],
                             use_container_width=True)
            else:
                st.info("暂无反馈记录")
        else:
            st.error("无法获取统计数据")

    except Exception as e:
        st.error(f"连接后端服务失败: {str(e)}")


def main():
    """主函数"""
    init_session_state()

    # 侧边栏导航
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面", ["问答", "评测面板"])

    # 页面路由
    if page == "问答":
        upload_handler()
        chat_interface()
    elif page == "评测面板":
        evaluation_dashboard()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **项目说明**

    - 基于 RAG 的智能问答系统
    - 支持文档上传和向量化
    - 集成工具调用功能
    - 包含评测面板
    """)


if __name__ == "__main__":
    main()
