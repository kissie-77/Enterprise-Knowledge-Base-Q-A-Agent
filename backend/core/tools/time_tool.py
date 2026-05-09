"""
时间工具 - 获取当前时间和日期
"""
from datetime import datetime
from ..tool_registry import BaseTool


class TimeTool(BaseTool):
    """获取当前时间和日期的工具"""

    name = "get_current_time"
    description = "获取当前的日期和时间。当用户询问现在几点、今天日期等时间相关问题时使用此工具。"

    async def execute(self, *args, **kwargs) -> str:
        now = datetime.now()
        return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} (星期{'一二三四五六日'[now.weekday()]})"
