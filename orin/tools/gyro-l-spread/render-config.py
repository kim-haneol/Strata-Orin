"""Expand documented environment placeholders into a new JSON file."""
import json
import os
from pathlib import Path
from string import Template
import sys


def expand(value):
    if isinstance(value, str):
        return Template(value).substitute(os.environ)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    result = expand(json.loads(source.read_text()))
    with destination.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
