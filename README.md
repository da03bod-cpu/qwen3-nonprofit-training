# Qwen3 Nonprofit Structured-JSON Fine-Tuning

This repository fine-tunes Qwen3-8B with LoRA/QLoRA for Arabic nonprofit annual-report extraction.

## Where the 20 files go

Put the 10 PaddleOCR-VL JSON inputs here:

`data/ocr/G_1.json` ... `data/ocr/G_10.json`

Put the 10 ground-truth output JSON files here:

`data/labels/G_1.json` ... `data/labels/G_10.json`

The files are paired by the same `G_N` name.

The OCR JSON can be the full PaddleOCR-VL response. The preparation script extracts `block_content` from `parsing_res_list` in page/order sequence, which matches the text representation used by the production OCR→Qwen pipeline.

## Build dataset

```bash
python scripts/prepare_dataset.py
```

This creates `data/train.json`, `data/validation.json`, and `data/test.json`.

## Train

```bash
python train.py --config configs/train.yaml
```

The default model is `Qwen/Qwen3-8B` and training is LoRA/QLoRA-friendly for a single 24GB+ NVIDIA GPU.

## Important

Do not put PDFs into the training folder. Training uses the completed PaddleOCR JSON outputs as input and the ground-truth program JSON as the assistant target.

## Current dataset status
The OCR inputs in `data/ocr/` are the actual PaddleOCR outputs from `OCR_Output.zip` (G_1 through G_10). Ground-truth labels are in `data/labels/`.

The prepared dataset is written to `data/train.json`, `data/validation.json`, and `data/test.json`.

Note: some reports are long enough that a 16k-token context may truncate the sample. Check token lengths with the Qwen3 tokenizer on the training GPU before starting the run. If necessary, raise `max_seq_length` or redesign the dataset around aligned document sections rather than silently truncating.

Training requires an NVIDIA GPU. The current ChatGPT execution environment has no NVIDIA GPU, so the training process itself has not been started here.
