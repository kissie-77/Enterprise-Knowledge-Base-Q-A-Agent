"""
网页搜索工具 - 支持 SerpAPI 和 DuckDuckGo
"""
import os
from typing import Optional
from ..tool_registry import BaseTool


class WebSearchTool(BaseTool):
    """网页搜索工具，当知识库中没有相关信息时使用"""

    name = "web_search"
    description = "搜索互联网获取最新信息。当知识库中没有找到答案，或用户询问实时信息（新闻、天气、最新事件等）时使用此工具。输入搜索关键词即可。"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("SERPAPI_API_KEY", "")

    async def execute(self, query: str = "", *args, **kwargs) -> str:
        """
        执行网页搜索

        Args:
            query: 搜索关键词
        """
        if not query and args:
            query = str(args[0])

        if not query:
            return "错误：请提供搜索关键词"

        # 如果配置了 SerpAPI
        if self._api_key:
            return await self._search_serpapi(query)

        # 降级：返回提示信息
        return f"搜索功能暂未配置 API 密钥。用户搜索的问题是: '{query}'。请基于已有知识回答。"

    async def _search_serpapi(self, query: str) -> str:
        """使用 SerpAPI 搜索"""
        try:
            from serpapi import SerpApiClient

            params = {
                "engine": "google",
                "q": query,
                "api_key": self._api_key,
                "gl": "cn",
                "hl": "zh-cn",
                "num": 5,
            }

            client = SerpApiClient(params)
            results = client.get_dict()

            # 智能解析结果（参考 hello-agents 的实现）
            if "answer_box" in results and "answer" in results["answer_box"]:
                return f"直接答案: {results['answer_box']['answer']}"

            if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                kg = results["knowledge_graph"]
                title = kg.get("title", "")
                desc = kg.get("description", "")
                return f"{title}: {desc}"

            if "organic_results" in results and results["organic_results"]:
                snippets = []
                for i, res in enumerate(results["organic_results"][:3]):
                    title = res.get("title", "")
                    snippet = res.get("snippet", "")
                    snippets.append(f"[{i+1}] {title}\n{snippet}")
                return "\n\n".join(snippets)

            return f"未找到关于 '{query}' 的相关信息。"

        except ImportError:
            return "搜索功能需要安装 serpapi 包: pip install google-search-results"
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"
