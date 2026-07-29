#!/usr/bin/env python3
import json
from pathlib import Path


def _evaluate(parameters: dict[str, dict[str, float]], t: float) -> float:
    return (
        parameters["a"]["value"] * t**2
        + parameters["b"]["value"] * t
        + parameters["c"]["value"]
    )


if __name__ == "__main__":
    parameters = json.loads(Path("parameters.json").read_text(encoding="utf-8"))
    output = [_evaluate(parameters, t) for t in range(10)]
    Path("poly.out").write_text("\n".join(map(str, output)), encoding="utf-8")
