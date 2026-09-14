import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR = ROOT / 'data' / 'ocr'
LABELS = ROOT / 'data' / 'labels'
OUT = ROOT / 'data'

SYSTEM = '''أنت نموذج متخصص في استخراج البرامج والمشاريع والمبادرات والأنشطة من التقارير السنوية للجمعيات غير الربحية.
اقرأ نص OCR كما هو، حتى لو احتوى على أخطاء OCR بسيطة. استخرج البرامج والمشاريع والأنشطة المذكورة فعليًا فقط.
أعد JSON فقط وفق المخطط المطلوب، ولا تضف شرحًا خارج JSON.
صحح أخطاء OCR الواضحة فقط عندما يكون التصحيح مؤكدًا من السياق، ولا تخترع معلومات غير موجودة.
'''

# Keep duplicate reports in the same split to avoid leakage.
SPLITS = {
    'train': ['G_1', 'G_10', 'G_2', 'G_3', 'G_4', 'G_5', 'G_6'],
    'validation': ['G_7', 'G_8'],
    'test': ['G_9'],
}

def ocr_to_text(obj):
    # OCR_Output.zip contains both the cleaned format used by our earlier
    # OCR Code node and the raw PaddleOCR RunPod format. Support both.
    if isinstance(obj, list):
        if not obj:
            return ''
        obj = obj[0]

    if not isinstance(obj, dict):
        return ''

    # Cleaned OCR format: {pages:[{page, blocks:[{order,label,text}], text:...}]}
    if isinstance(obj.get('full_text'), str) and obj.get('full_text').strip():
        return obj['full_text'].strip()

    result = obj.get('output', {}).get('result', obj.get('result', {}))
    pages = result.get('pages', []) if isinstance(result, dict) else []
    page_parts = []
    for page in pages:
        pno = page.get('page')
        res = page.get('data', {}).get('res', {})
        blocks = res.get('parsing_res_list', []) or []
        ordered = []
        for block in blocks:
            text = block.get('block_content')
            if not isinstance(text, str) or not text.strip():
                continue
            order = block.get('block_order')
            order_key = order if isinstance(order, (int, float)) else 10**9
            ordered.append((order_key, text.strip()))
        ordered.sort(key=lambda x: x[0])
        page_text = '\n'.join(text for _, text in ordered)
        if page_text:
            page_parts.append(f'===== PAGE {pno} =====\n{page_text}')
    return '\n\n'.join(page_parts)

def normalize_label(obj):
    if isinstance(obj, list):
        programs = obj
    elif isinstance(obj, dict):
        programs = (
            obj.get('programs_and_projects')
            or obj.get('programs')
            or obj.get('projects')
            or []
        )
    else:
        programs = []
    return {'programs': programs}

def make_sample(name):
    ocr_path = OCR / f'{name}.json'
    label_path = LABELS / f'{name}.json'
    if not ocr_path.exists():
        raise FileNotFoundError(f'Missing OCR file: {ocr_path}')
    if not label_path.exists():
        raise FileNotFoundError(f'Missing label file: {label_path}')

    ocr = json.loads(ocr_path.read_text(encoding='utf-8'))
    if isinstance(ocr, dict) and ocr.get('_PLACEHOLDER'):
        raise RuntimeError(f'Missing completed PaddleOCR JSON: {ocr_path}')

    label = normalize_label(json.loads(label_path.read_text(encoding='utf-8')))
    text = ocr_to_text(ocr)
    if not text:
        raise RuntimeError(f'No OCR text found in {ocr_path}')

    target = json.dumps(label, ensure_ascii=False, separators=(',', ':'))
    return {
        'id': name,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': text},
            {'role': 'assistant', 'content': target},
        ],
    }

if __name__ == '__main__':
    for split, names in SPLITS.items():
        samples = [make_sample(name) for name in names]
        path = OUT / f'{split}.json'
        path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{split}: {len(samples)} samples -> {path}')
