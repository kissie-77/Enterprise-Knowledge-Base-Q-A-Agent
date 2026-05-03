"""
工具函数模块
"""
from datetime import datetime
import ast
import operator


def get_current_time() -> str:
    """
    获取当前时间

    Returns:
        格式化的时间字符串，如 "2024-01-15 14:30:25"
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    安全计算数学表达式

    Args:
        expression: 数学表达式字符串，如 "2+3*4"

    Returns:
        计算结果字符串
    """
    # 定义允许的操作符
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # Python 3.8 之前
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return operators[type(node.op)](operand)
        else:
            raise TypeError(f"不支持的表达式类型: {type(node)}")

    try:
        # 解析表达式
        tree = ast.parse(expression, mode='eval')

        # 只允许包含数字、运算符和括号的表达式
        for node in ast.walk(tree):
            if not isinstance(node, (
                ast.Expression, ast.BinOp, ast.UnaryOp,
                ast.Num, ast.Constant, ast.Operator,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd
            )):
                raise ValueError("表达式包含不支持的操作")

        result = _eval(tree.body)

        # 格式化结果
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具注册表
TOOLS = {
    "get_current_time": {
        "description": "获取当前时间",
        "parameters": {}
    },
    "calculate": {
        "description": "计算数学表达式",
        "parameters": {
            "expression": "数学表达式字符串"
        }
    }
}


def get_tools_info() -> dict:
    """获取所有工具的信息"""
    return TOOLS
