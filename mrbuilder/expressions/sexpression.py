import re

_EXPRESSION_START = "{{"
_EXPRESSION_END = "}}"

_MATCH_EXPRESSION = re.compile("^" + _EXPRESSION_START + ".*" + _EXPRESSION_END + "$")


def eval_expression(expression, variable_resolver):
    if _is_expression(expression):
        return _eval_and_parse_expression(expression, variable_resolver)
    else:
        return variable_resolver(expression)


def _is_expression(expression):
    return isinstance(expression, str) and \
           re.match(_MATCH_EXPRESSION, str(expression))


def _eval_and_parse_expression(expression, variable_resolver):
    expression = expression.replace(_EXPRESSION_START, "")\
                           .replace(_EXPRESSION_END, "")
    split_expression = expression.split()
    if len(split_expression) == 3:
        left, oper, right = split_expression
        left, converted = _convert(left)
        if not converted:
            left = variable_resolver(left)

        right, converted = _convert(right)
        if not converted:
            right = variable_resolver(right)

        return _eval_ternary(left, oper, right)
    else:
        raise ValueError("Unsupported Expression must have 3 parts separated by spaces")


def _eval_ternary(left, middle, right):
    if middle == "*":
        return left * right
    elif middle == "+":
        return left + right
    elif middle == "-":
        return left - right
    elif middle == "/":
        return left / right
    elif middle == "%":
        return left % right
    elif middle == "**":
        return left ** right
    else:
        raise ValueError("Unsupported Operator: " + middle)


def _convert(operand):
    converted, successful = _convert_int(operand)
    if not successful:
        converted, successful = _convert_float(operand)

    return converted, successful


def _convert_float(operand):
    try:
        return float(operand), True
    except ValueError:
        return operand, False


def _convert_int(operand):
    try:
        return int(operand), True
    except ValueError:
        return operand, False
