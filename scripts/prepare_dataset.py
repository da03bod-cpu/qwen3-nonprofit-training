cd /workspace/qwen3-nonprofit-training

cat > scripts/prepare_dataset.py <<'PY'
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR_DIR = ROOT / "data" / "ocr"
LABEL_DIR = ROOT / "data" / "labels"
OUT_DIR = ROOT / "data" / "prepared"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": ["G_1", "G_2", "G_3", "G_4", "G_5", "G_6", "G_10"],
    "validation": ["G_7", "G_8"],
    "test": ["G_9"],
}

MAX_CHARS = 45000


def load_ocr(name):
    data = json.load(open(OCR_DIR / f"{name}.json", encoding="utf-8"))

    root = data[0] if isinstance(data, list) else data

    pages = []

    for page in root.get("pages", []):
        page_num = page.get("page")
        blocks = page.get("blocks", [])

        texts = []

        for block in blocks:
            text = block.get("text", "")
            if not isinstance(text, str):
                continue

            text = " ".join(text.split()).strip()

            if text:
                texts.append(text)

        if texts:
            pages.append({
                "page": page_num,
                "text": "\n".join(texts)
            })

    return pages


def load_labels(name):
    data = json.load(open(LABEL_DIR / f"{name}.json", encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("programs_and_projects", [])

    return []


def make_output(name, text, labels):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "أنت نموذج متخصص في استخراج البرامج والمشاريع من التقارير السنوية "
                    "للجمعيات غير الربحية. استخرج البيانات المطلوبة من النص وأخرج JSON فقط."
                )
            },
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    labels,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
            }
        ],
        "source": name
    }


for split, names in SPLITS.items():

    examples = []

    for name in names:

        pages = load_ocr(name)
        labels = load_labels(name)

        # ---------------------------------------------------------
        # نبدأ بتجميع الصفحات في chunks متوسطة الحجم.
        # لا نقطع الصفحة نفسها.
        # ---------------------------------------------------------

        current = []
        current_len = 0

        for page in pages:

            page_text = f"===== PAGE {page['page']} =====\n{page['text']}"
            page_len = len(page_text)

            if current and current_len + page_len > MAX_CHARS:
                text = "\n\n".join(current)

                examples.append(
                    make_output(name, text, labels)
                )

                current = []
                current_len = 0

            current.append(page_text)
            current_len += page_len

        if current:
            text = "\n\n".join(current)

            examples.append(
                make_output(name, text, labels)
            )

    output = OUT_DIR / f"{split}.jsonl"

    with open(output, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"{split}: {len(examples)} examples")
    print(f"saved: {output}")

PY
