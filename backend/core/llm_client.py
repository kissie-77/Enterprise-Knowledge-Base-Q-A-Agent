"""
通用 LLM 客户端模块
兼容任何 OpenAI 接口的服务（智谱、DeepSeek、通义千问、Moonshot 等）
参考 hello-agents 的 HelloAgentsLLM 设计
"""
import os
from typing import List, Dict, Optional, AsyncIterator
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


# 预设的 Provider 配置
PROVIDER_CONFIGS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "env_key": "ZHIPU_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "env_key": "QWEN_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "env_key": "MOONSHOT_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "silicon": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "env_key": "SILICON_API_KEY",
    },
}


class LLMClient:
    """
    通用 LLM 客户端
    支持任何兼容 OpenAI Chat Completions 接口的服务

    使用方式:
        # 方式1: 使用预设 provider
        llm = LLMClient(provider="zhipu")

        # 方式2: 自定义配置
        llm = LLMClient(
            api_key="your-key",
            base_url="https://your-api-endpoint/v1",
            model="your-model"
        )

        # 方式3: 从环境变量自动加载
        llm = LLMClient()  # 读取 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        """
        初始化 LLM 客户端

        Args:
            provider: 预设 provider 名称（zhipu/deepseek/qwen/moonshot/openai/silicon）
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            temperature: 生成温度
            max_tokens: 最大 token 数
            timeout: 请求超时时间（秒）
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 解析配置：provider > 显式参数 > 环境变量
        if provider and provider in PROVIDER_CONFIGS:
            config = PROVIDER_CONFIGS[provider]
            self.base_url = base_url or config["base_url"]
            self.model = model or config["default_model"]
            self.api_key = api_key or os.getenv(config["env_key"], "")
        else:
            self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
            self.model = model or os.getenv("LLM_MODEL", "glm-4-flash")
            self.api_key = api_key or os.getenv("LLM_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                f"API 密钥未配置。请设置环境变量或直接传入 api_key 参数。\n"
                f"Provider: {provider or '自定义'}"
            )

        # 创建异步 OpenAI 客户端
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        print(f"🧠 LLM 客户端初始化完成: model={self.model}, base_url={self.base_url}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        调用 LLM 进行对话（非流式）

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大 token（覆盖默认值）

        Returns:
            模型回复内容
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=False,
                **kwargs,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        流式调用 LLM

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 生成温度
            max_tokens: 最大 token

        Yields:
            文本片段
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

        except Exception as e:
            print(f"❌ LLM 流式调用失败: {e}")
            raise

    async def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
    ) -> str:
        """
        简化的思考接口（参考 hello-agents 的 think 方法）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 生成温度，默认 0（确定性输出）

        Returns:
            模型回复
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(messages, temperature=temperature)

    def __repr__(self):
        return f"<LLMClient(model='{self.model}', base_url='{self.base_url}')>"
