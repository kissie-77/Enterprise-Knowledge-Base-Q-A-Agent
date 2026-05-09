"""
工具注册与执行模块
参考 hello-agents 的 ToolExecutor 设计，提供模块化工具管理

使用方式:
    # 方式1: 使用装饰器注册工具函数
    registry = ToolRegistry()

    @registry.tool(name="search", description="搜索网页")
    def search(query: str) -> str:
        return "搜索结果..."

    # 方式2: 继承 BaseTool 创建工具类
    class CalculatorTool(BaseTool):
        name = "calculator"
        description = "计算数学表达式"

        async def execute(self, expression: str, **kwargs) -> str:
            return str(eval(expression))

    registry.register(CalculatorTool())

    # 方式3: 直接注册函数
    registry.register_function("get_time", "获取当前时间", get_time_func)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None


class BaseTool(ABC):
    """
    工具基类
    所有自定义工具都应继承此类
    """
    name: str = "base_tool"
    description: str = "基础工具"

    def __init__(self):
        if not hasattr(self, 'parameters'):
            self.parameters: List[ToolParameter] = []

    @abstractmethod
    async def execute(self, *args, **kwargs) -> str:
        """
        执行工具

        Returns:
            工具执行结果（字符串）
        """
        pass

    def get_description(self) -> str:
        """获取工具描述（用于 prompt 构建）"""
        params_desc = ""
        if hasattr(self, 'parameters') and self.parameters and isinstance(self.parameters, list):
            params_list = [f"{p.name}({p.type}): {p.description}" for p in self.parameters]
            params_desc = f" 参数: {', '.join(params_list)}"
        return f"{self.name}: {self.description}{params_desc}"


class FunctionTool(BaseTool):
    """
    基于函数的工具封装
    将普通函数包装为 BaseTool 接口
    """

    def __init__(self, name: str, description: str, func: Callable, is_async: bool = False):
        self.name = name
        self.description = description
        self._func = func
        self._is_async = is_async
        self.parameters = []

    async def execute(self, *args, **kwargs) -> str:
        """执行包装的函数"""
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

    参考 hello-agents 的 ToolExecutor 模式
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """
        注册一个工具实例

        Args:
            tool: BaseTool 实例

        Returns:
            self（支持链式调用）
        """
        if tool.name in self._tools:
            print(f"⚠️ 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"🔧 工具 '{tool.name}' 已注册: {tool.description}")
        return self

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable,
        is_async: bool = False,
    ) -> "ToolRegistry":
        """
        注册一个函数作为工具

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数
            is_async: 是否为异步函数

        Returns:
            self（支持链式调用）
        """
        tool = FunctionTool(name=name, description=description, func=func, is_async=is_async)
        return self.register(tool)

    def tool(self, name: str, description: str, is_async: bool = False):
        """
        装饰器方式注册工具

        Usage:
            @registry.tool(name="search", description="搜索工具")
            def search(query: str) -> str:
                ...
        """
        def decorator(func: Callable):
            self.register_function(name, description, func, is_async)
            return func
        return decorator

    def unregister(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            print(f"🔧 工具 '{name}' 已注销")
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)

    async def execute_tool(self, name: str, *args, **kwargs) -> str:
        """
        执行指定工具

        Args:
            name: 工具名称
            *args, **kwargs: 传递给工具的参数

        Returns:
            工具执行结果
        """
        tool = self._tools.get(name)
        if not tool:
            return f"错误：未找到名为 '{name}' 的工具。可用工具: {', '.join(self._tools.keys())}"

        try:
            result = await tool.execute(*args, **kwargs)
            return result
        except Exception as e:
            return f"工具 '{name}' 执行失败: {str(e)}"

    def get_tools_description(self) -> str:
        """
        获取所有已注册工具的格式化描述
        用于构建 Agent 的 prompt

        Returns:
            工具描述字符串
        """
        if not self._tools:
            return "暂无可用工具"

        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(f"- {tool.get_description()}")
        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """列出所有注册的工具名称"""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self):
        tools_str = ", ".join(self._tools.keys())
        return f"<ToolRegistry(tools=[{tools_str}])>"
