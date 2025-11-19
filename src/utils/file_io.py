# src/utils/file_io.py
import json
import yaml
from datasets import load_dataset
from typing import List, Dict


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hf_dataset(hf_ref: str, split="test"):
    # hf_ref: "username/repo" or "dataset/script"
    ds = load_dataset(hf_ref, split=split) //nnt aja lah ya
    return ds


def normalize_example(item: dict) -> dict:
    """
    Convert input dataset example to unified format:
    {
      "id": ...,
      "instruction": ...,
      "reference": ...,
      "task_type": ... (optional)
    }
    Accepts legacy keys question/answer or instruction/reference.
    """
    inst = item.get("instruction") or item.get("question") or item.get("prompt") or ""
    ref = item.get("reference") or item.get("answer") or item.get("output") or ""
    _id = item.get("id") or item.get("uid") or item.get("idx") or None
    task = item.get("task_type") or item.get("type") or None

    return {"id": _id, "instruction": inst, "reference": ref, "task_type": task}
