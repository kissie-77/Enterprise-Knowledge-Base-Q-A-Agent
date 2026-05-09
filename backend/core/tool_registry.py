"""
工具注册与执行模块
参考 hello-agents 的 ToolExecutor 设计，提供模块化工具管理
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List


class BaseTool(ABC):
    """
    工具基类
    所有自定义工具都应继承此类
    """
    name: str = "base_tool"
    description: str = "基础工具"

    @abstractmethod
    async def execute(self, *args, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        pass

    def get_description(self) -> str:
        """获取工具描述（用于 prompt 构建）"""
        return f"{self.name}: {self.description}"


class FunctionTool(BaseTool):
    """基于函数的工具封装"""

    def __init__(self, name: str, description: str, func: Callable, is_async: bool = False):
        self.name = name
        self.description = description
        self._func = func
        self._is_async = is_async

    async def execute(self, *args, **kwargs) -> str:
        try:
            if self._is_async:
                result = await self._func(*args, **kwargs)
            else:
                result = self._func(*args, **kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


class ToolRegistry:
    """
    工具注册表
    管理所有可用工具的注册、查找和执行
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """注册一个工具实例"""
        if tool.name in self._tools:
            print(f"⚠️ 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"🔧 工具 '{tool.name}' 已注册: {tool.description}")
        return self

    def register_function(
        self, name: str, description: str, func: Callable, is_async: bool = False,
    ) -> "ToolRegistry":
        """注册一个函数作为工具"""
        tool = FunctionTool(name=name, description=description, func=func, is_async=is_async)
        return self.register(tool)

    def tool(self, name: str, description: str, is_async: bool = False):
        """装饰器方式注册工具"""
        def decorator(func: Callable):
            self.register_function(name, description, func, is_async)
            return func
        return decorator

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    async def execute_tool(self, name: str, *args, **kwargs) -> str:
        """执行指定工具"""
        tool = self._tools.get(name)
        if not tool:
            return f"错误：未找到名为 '{name}' 的工具。可用工具: {', '.join(self._tools.keys())}"
        try:
            result = await tool.execute(*args, **kwargs)
            return result
        except Exception as e:
            return f"工具 '{name}' 执行失败: {str(e)}"

    def get_tools_description(self) -> str:
        """获取所有工具的格式化描述"""
        if not self._tools:
            return "暂无可用工具"
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(f"- {tool.get_description()}")
        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self):
        tools_str = ", ".join(self._tools.keys())
        return f"<ToolRegistry(tools=[{tools_str}])>"
