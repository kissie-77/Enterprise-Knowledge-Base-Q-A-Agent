"""
智谱 AI (Zhipu) LLM 客户端
"""
import os
import httpx
from typing import Dict, Any

# 从环境变量获取 API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4"


async def call_zhipu_llm(prompt: str, model: str = "glm-4-flash") -> str:
    """
    调用智谱 AI API

    Args:
        prompt: 提示词
        model: 模型名称，默认使用 glm-4-flash（免费）

    Returns:
        模型回复内容
    """
    if not ZHIPU_API_KEY:
        raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ZHIPU_API_URL}/chat/completions",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "API 返回格式异常"
            else:
                error_msg = f"API 调用失败: {response.status_code} - {response.text}"
                print(error_msg)
                return f"抱歉，处理请求时出现错误: {response.status_code}"

    except httpx.TimeoutException:
        return "请求超时，请稍后重试"
    except Exception as e:
        print(f"调用 LLM 时发生错误: {e}")
        return f"处理请求时发生错误: {str(e)}"


async def call_zhipu_llm_stream(prompt: str, model: str = "glm-4-flash"):
    """
    流式调用智谱 AI API（生成器）

    Args:
        prompt: 提示词
        model: 模型名称

    Yields:
        生成的文本片段
    """
    if not ZHIPU_API_KEY:
        raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{ZHIPU_API_URL}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # 移除 "data: " 前缀
                            if data == "[DONE]":
                                break
                            try:
                                chunk = httpx.json.decode(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except:
                                pass
                else:
                    yield f"API 调用失败: {response.status_code}"
    except Exception as e:
        yield f"流式处理错误: {str(e)}"
