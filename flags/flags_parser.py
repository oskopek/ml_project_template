import json
import re
from dotmap import DotMap

FLAGS = None


def parse(file_name, remainder):
    lines = None
    with open(file_name, mode='r') as f:
        lines = f.readlines()

    input_str = re.sub(r'\\\n', '', "\n".join(lines))
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)

    # TODO: Incorporate the remainder (arg overrides)

    global FLAGS
    FLAGS = DotMap(data)
