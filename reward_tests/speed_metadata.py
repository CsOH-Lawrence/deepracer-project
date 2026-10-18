"""Pure metadata transformation; no network/filesystem or model-weight edits."""
import copy
import math


def scale_all_speeds(metadata, factor):
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Speed factor must be finite and positive")
    result = copy.deepcopy(metadata)
    if not result.get("action_space") or not isinstance(result["action_space"], list):
        raise ValueError("Expected nonempty discrete action list")
    for action in result["action_space"]:
        original = float(action["speed"])
        if not math.isfinite(original) or original <= 0:
            raise ValueError("Action speed must be finite and positive")
        scaled = round(original * factor, 6)
        if not math.isfinite(scaled) or scaled <= 0:
            raise ValueError("Invalid scaled action speed")
        action["speed"] = scaled
    return result
