import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


SOURCE_TARGET_BY_DIRECTION = {
    "t1t2": ("T1", "T2_FLAIR", "t1tot2"),
    "t2t1": ("T2_FLAIR", "T1", "t2tot1"),
}


def _parse_scalar(value):
    value = value.strip()
    lower = value.lower()
    if lower in {"null", "none", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value.startswith("[") or value.startswith("{"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text):
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        item = line.strip()
        if ":" not in item:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_config(path):
    path = Path(path)
    text = path.read_text()
    try:
        import yaml

        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _simple_yaml_load(text)
    if cfg is None:
        cfg = {}
    cfg["_config_path"] = str(path)
    return cfg


def deep_update(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(cfg, args):
    cfg = deepcopy(cfg)
    if getattr(args, "direction", None):
        cfg.setdefault("task", {})["direction"] = args.direction
    if getattr(args, "device", None):
        cfg.setdefault("runtime", {})["device"] = args.device
    if getattr(args, "resume", None):
        cfg.setdefault("training", {})["resume"] = args.resume
    return cfg


def parse_cli_args(specs, argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    values = {dest: default for _, dest, default, _ in specs}
    by_flag = {flag: (dest, choices) for flag, dest, _, choices in specs}
    idx = 0
    while idx < len(args):
        flag = args[idx]
        if flag not in by_flag:
            raise SystemExit(f"Unknown argument: {flag}")
        if idx + 1 >= len(args):
            raise SystemExit(f"Missing value for {flag}")
        value = args[idx + 1]
        dest, choices = by_flag[flag]
        if choices is not None and value not in choices:
            raise SystemExit(f"Invalid value for {flag}: {value}. Expected one of: {', '.join(choices)}")
        values[dest] = value
        idx += 2
    if values.get("config") is None:
        raise SystemExit("Missing required argument: --config")
    return SimpleNamespace(**values)


def direction_spec(cfg):
    direction = str(cfg.get("task", {}).get("direction", "t1t2")).lower()
    if direction not in SOURCE_TARGET_BY_DIRECTION:
        valid = ", ".join(sorted(SOURCE_TARGET_BY_DIRECTION))
        raise ValueError(f"Unknown task.direction={direction!r}. Valid values: {valid}.")
    source, target, output_subdir = SOURCE_TARGET_BY_DIRECTION[direction]
    return {
        "direction": direction,
        "source": source,
        "target": target,
        "output_subdir": output_subdir,
    }


def split_run_id(cfg):
    mode = str(cfg.get("split", {}).get("mode", "seed42")).lower()
    if mode != "seed42":
        raise ValueError("Only seed42 split is supported.")
    return "seed42"


def model_family(cfg):
    family = cfg.get("model", {}).get("family")
    if family:
        return str(family).lower()
    name = str(cfg.get("model", {}).get("name", "model")).lower()
    if "3d" in name and "unet" in name:
        return "3dunet"
    return name.replace(" ", "_")


def get_output_dir(cfg):
    spec = direction_spec(cfg)
    root = Path(cfg.get("output", {}).get("root", "output"))
    variant = cfg.get("output", {}).get("variant")
    if variant:
        return root / model_family(cfg) / str(variant).lower() / spec["output_subdir"]
    return root / model_family(cfg) / spec["output_subdir"]


def list_int(value, name):
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list.")
    return [int(v) for v in value]
