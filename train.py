import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(
        'json',
        data_files={
            'train': 'data/train.json',
            'validation': 'data/validation.json',
            'test': 'data/test.json',
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model_name'], use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg['model_name'],
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['target_modules'],
        bias='none',
        task_type='CAUSAL_LM',
    )

    training_args = SFTConfig(
        output_dir=cfg['output_dir'],
        num_train_epochs=cfg['num_train_epochs'],
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],
        warmup_ratio=cfg['warmup_ratio'],
        logging_steps=cfg['logging_steps'],
        save_steps=cfg['save_steps'],
        save_total_limit=cfg['save_total_limit'],
        bf16=cfg['bf16'],
        fp16=cfg['fp16'],
        gradient_checkpointing=cfg['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none',
        eval_strategy='steps',
        eval_steps=cfg['save_steps'],
        seed=cfg['seed'],
        max_length=cfg['max_seq_length'],
        packing=cfg['packing'],
        assistant_only_loss=True,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(cfg['output_dir'])
    tokenizer.save_pretrained(cfg['output_dir'])
    print(f'LoRA adapter saved to: {cfg["output_dir"]}')


if __name__ == '__main__':
    main()
