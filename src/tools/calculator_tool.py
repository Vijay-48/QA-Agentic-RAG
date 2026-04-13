"""Calculator tool — safe mathematical expression evaluation.

Uses Python's ast.literal_eval and a restricted evaluator instead of
raw eval() to prevent code injection.
"""
from __future__ import annotations

import ast
import operator
import math

from src.core.base_tool import BaseTool
from src.core.logger import get_logger

logger = get_logger("CalculatorTool")

# Supported binary operators
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

# Safe math functions the calculator can use
_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only arithmetic operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    elif isinstance(node, ast.Call):
        # Support safe math functions like sqrt(16)
        if isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS:
            args = [_safe_eval(arg) for arg in node.args]
            return float(_MATH_FUNCS[node.func.id](*args))
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    """Perform mathematical calculations safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform mathematical calculations. "
            "Input should be a mathematical expression like '15 * 3 + 7' or 'sqrt(144)'. "
            "Supports: +, -, *, /, **, sqrt, abs, round, min, max. "
            "Use this when you need to do math operations."
        )

    def execute(self, input_text: str) -> str:
        """Evaluate a mathematical expression safely.

        Args:
            input_text: A math expression string.

        Returns:
            The result as a string, or an error message.
        """
        expression = input_text.strip()
        logger.info("Calculating: '%s'", expression)

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            # Format nicely: remove trailing .0 for integers
            if result == int(result):
                formatted = str(int(result))
            else:
                formatted = f"{result:.6f}".rstrip("0").rstrip(".")
            logger.info("Result: %s = %s", expression, formatted)
            return f"{expression} = {formatted}"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except (ValueError, TypeError, SyntaxError) as e:
            return f"Error: Could not evaluate '{expression}'. {e}"
