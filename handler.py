import json
import os
import subprocess
import sys
from pathlib import Path

import runpod

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "train.yaml"


def _env_setup():
    # Keep Hugging Face cache and training outputs on the persistent RunPod volume.
    volume = Path(os.environ.get("RUNPOD_VOLUME", "/runpod-volume"))
    hf_home = volume / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    (volume / "outputs").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    # Avoid the Xet download path; regular HTTP is more robust on ephemeral workers.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _run(command):
    print("[serverless] running:", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        lines.append(line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Training command failed with exit code {return_code}. "
            f"Last log lines: {lines[-30:]}"
        )
    return lines


def _audit():
    # Exact tokenizer audit before training; this avoids silently truncating examples.
    from transformers import AutoTokenizer

    with open(CONFIG, encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], use_fast=False, trust_remote_code=True
    )
    report = {}
    for split in ("train", "validation", "test"):
        path = ROOT / f"data/{split}.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for row in records:
            messages = row.get("messages", [])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            rows.append({"id": row.get("id"), "tokens": token_count})
        report[split] = rows

    limit = int(cfg["max_seq_length"])
    over = [r for rows in report.values() for r in rows if r["tokens"] > limit]
    return {
        "model": cfg["model_name"],
        "max_seq_length": limit,
        "over_limit": over,
        "report": report,
    }


def handler(job):
    _env_setup()
    inp = job.get("input", {})
    action = inp.get("action", "audit")

    if action == "audit":
        return _audit()

    if action != "train":
        return {"error": f"Unknown action: {action}. Use 'audit' or 'train'."}

    config = inp.get("config", str(CONFIG))
    command = [sys.executable, str(ROOT / "train.py"), "--config", config]
    logs = _run(command)

    output_dir = os.environ.get(
        "OUTPUT_DIR", "/runpod-volume/outputs/qwen3-nonprofit-lora"
    )
    return {
        "status": "completed",
        "output_dir": output_dir,
        "last_log_lines": logs[-20:],
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
