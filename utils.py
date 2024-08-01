import re
import torch


def get_slot_value(slot_value: str) -> str:
    groups = re.search(r"(SLOT_\w+)", slot_value)

    if not groups:
        return ""

    return groups.group(1)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
