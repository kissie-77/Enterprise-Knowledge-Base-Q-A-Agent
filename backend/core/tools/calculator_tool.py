"""
计算器工具 - 安全计算数学表达式
"""
import ast
import operator
from ..tool_registry import BaseTool


class CalculatorTool(BaseTool):
    """安全的数学计算工具"""

    name = "calculator"
    description = "计算数学表达式。输入一个数学表达式（如 '2+3*4' 或 '(100-20)/5'），返回计算结果。支持加减乘除和幂运算。"

    # 允许的运算符
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    async def execute(self, expression: str = "", *args, **kwargs) -> str:
        """
        安全计算数学表达式

        Args:
            expression: 数学表达式字符串
        """
        # 如果 expression 为空，尝试从 args 获取
        if not expression and args:
            expression = str(args[0])

        if not expression:
            return "错误：请提供数学表达式"

        # 清理表达式
        expression = expression.strip()

        try:
            tree = ast.parse(expression, mode='eval')

            # 安全检查：只允许数字和运算符
            for node in ast.walk(tree):
                if not isinstance(node, (
                    ast.Expression, ast.BinOp, ast.UnaryOp,
                    ast.Num, ast.Constant, ast.operator,
                    ast.Add, ast.Sub, ast.Mult, ast.Div,
                    ast.Pow, ast.Mod, ast.FloorDiv,
                    ast.USub, ast.UAdd,
                )):
                    raise ValueError(f"不支持的操作: {type(node).__name__}")

            result = self._eval_node(tree.body)

            # 格式化结果
            if isinstance(result, float) and result.is_integer():
                return f"{expression} = {int(result)}"
            elif isinstance(result, float):
                return f"{expression} = {result:.6g}"
            return f"{expression} = {result}"

        except ZeroDivisionError:
            return "错误：除数不能为零"
        except ValueError as e:
            return f"错误：{str(e)}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def _eval_node(self, node):
        """递归求值 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 兼容
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op(operand)
        else:
            raise ValueError(f"不支持的表达式节点: {type(node).__name__}")
