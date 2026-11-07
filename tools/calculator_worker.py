"""Isolated process entry point for calculator evaluation."""

import json
import sys

from mylocalmodel.tools.calculator import _evaluate_expression


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        result = _evaluate_expression(request["expression"])
        response = {"ok": True, "result": result}
    except BaseException as exc:
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
