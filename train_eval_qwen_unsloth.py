import os
import re
import json
import time
import math
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
TRAIN_CSV = "TrainDataExtract.csv"
TEST_CSV  = "TestDataExtract.csv"

# name of HF / unsloth base
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
# how we’ll save our LoRA/adapter
SAVE_NAME = "Qwen2.5-7B_4bit_FINE"

MAX_SEQ_LENGTH = 6000
MAX_STEPS = 10_000  # your original
BATCH_SIZE = 1


# ---------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)


# ---------------------------------------------------------------------
# load model in 4bit + add LoRA
# ---------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = BASE_MODEL,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,
    load_in_4bit    = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

EOS_TOKEN = tokenizer.eos_token

alpaca_prompt = """Below is an  input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:
{}"""


def formatting_prompts_train(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    instructions = examples["Input"]
    outputs      = examples["True_output"]
    # we don’t really have a second “input” field, so we reuse instruction as input
    inputs       = examples.get("input", [""] * len(instructions))
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(inst, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


train_dataset = Dataset.from_pandas(train_df).map(
    formatting_prompts_train,
    batched=True,
)


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 2,
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = MAX_STEPS,
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer.train()


# ---------------------------------------------------------------------
# SAVE trained adapter + tokenizer
# ---------------------------------------------------------------------
save_dir = f"saved_{SAVE_NAME}"
os.makedirs(save_dir, exist_ok=True)

# this will save the PEFT/LoRA weights
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"[INFO] Model + tokenizer saved to: {save_dir}")


# ---------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------
def extract_json_model_output(text: str) -> Optional[Dict[str, Any]]:
    # drop <think>...</think>
    cleaned_text = text.replace("<think>", "").replace("</think>", "")
    resp_start = cleaned_text.find("### Response:")
    if resp_start != -1:
        response_text = cleaned_text[resp_start + len("### Response:"):].strip()
        json_start = response_text.find("{")
        if json_start != -1:
            json_block = response_text[json_start:]
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                return None
        return None
    # fallback: first {...}
    try:
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return None


def extract_json_true(text: str) -> Optional[Dict[str, Any]]:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def clean_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.replace("%", "").replace("L", "").strip()
        if value.lower() == "nan":
            return None
    return value


def is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", "null", "", "none"):
        return True
    if isinstance(value, np.ndarray):
        return np.isnan(value).all()
    return False


def is_numeric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)) and not math.isnan(value):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def compare_values(pred: Any, true: Any) -> bool:
    pred = clean_value(pred)
    true = clean_value(true)
    if is_nan(pred) and is_nan(true):
        return True
    try:
        return math.isclose(float(pred), float(true), rel_tol=1e-6)
    except (ValueError, TypeError):
        return str(pred) == str(true)


# ---------------------------------------------------------------------
# build test prompts
# ---------------------------------------------------------------------
alpaca_prompt_test = """Below is an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}
### Response:
""" + EOS_TOKEN


def formatting_prompts_test(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    instructions = examples["Input"]
    inputs       = examples.get("Input", [""] * len(instructions))
    texts = []
    for inst, inp in zip(instructions, inputs):
        texts.append(alpaca_prompt_test.format(inst, inp))
    return {"text": texts}


test_dataset = Dataset.from_pandas(test_df).map(
    formatting_prompts_test,
    batched=True,
)

# make model generation-only
FastLanguageModel.for_inference(model)

json_filename    = f"{SAVE_NAME}_eval.json"
metrics_filename = f"{SAVE_NAME}_metrics.txt"

if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            eval_results = json.load(f)
        except json.JSONDecodeError:
            eval_results = []
else:
    eval_results = []

TP = TN = FP = FN = 0
Total = Correct = 0
call_times: List[float] = []
do_sample = True

for idx, row in enumerate(test_dataset):
    prompt = row["text"]
    true_output = extract_json_true(test_df.loc[idx, "True_output"])

    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=MAX_SEQ_LENGTH,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    elapsed = time.time() - start
    call_times.append(elapsed)

    pred_json = extract_json_model_output(generated_text)

    if true_output is not None:
        Total += len(true_output.keys())
        for key, true_val in true_output.items():
            pred_val = pred_json.get(key) if pred_json else None
            if compare_values(pred_val, true_val):
                Correct += 1
                if is_numeric(true_val):
                    TP += 1
                else:
                    TN += 1
            else:
                if is_numeric(true_val):
                    FN += 1
                else:
                    FP += 1

    eval_results.append(
        {
            "index": idx,
            "prompt": prompt,
            "generated_text": generated_text,
            "true_output": true_output,
            "time": elapsed,
        }
    )

    with open(json_filename, "w") as f:
        json.dump(eval_results, f, indent=4)

avg_time = sum(call_times) / len(call_times) if call_times else 0.0
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

metrics_text = (
    f"Global Metrics:\n"
    f"Total Correct Predictions: {Correct}/{Total}\n"
    f"Total Correct Predictions %: {Correct / Total:.2%}\n"
    f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall:    {recall:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
    f"Average Call Time: {avg_time:.4f} seconds\n"
)

print(metrics_text)
with open(metrics_filename, "w") as f:
    f.write(metrics_text)

print(f"[INFO] eval JSON written to: {json_filename}")
print(f"[INFO] metrics written to:   {metrics_filename}")
