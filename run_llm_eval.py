import os
import re
import json
import time
import math
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI


# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
DATA_FILES = {
    "zero": "TestZero.csv",
    "one": "TestOne.csv",
    "few": "TestFew.csv",
}

# activate whichever ollama-style models you want to benchmark
MODELS = [
    "llama3.3:70b-instruct-fp16",
    # "llama3.3:70b-instruct-q8_0",
    # "llama3.1:70b-instruct-fp16",
]

# try multiple temperatures
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0]

# where your local ollama-compatible server is running
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11484/v1/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "R")  # or set to "" if not needed

MAX_GEN_LEN = 1900  # not used directly here but kept for clarity


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull the first {...} block from a model response."""
    if not text:
        return None
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def clean_value(value: Any) -> Any:
    """Remove common clinical suffixes like %/L and turn 'nan' into None."""
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
    if isinstance(value, str) and value.lower() == "nan":
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
    """Compare two fields with small numeric tolerance."""
    pred = clean_value(pred)
    true = clean_value(true)

    if is_nan(pred) and is_nan(true):
        return True

    try:
        return math.isclose(float(pred), float(true), rel_tol=1e-6)
    except (ValueError, TypeError):
        return str(pred) == str(true)


def load_dataset_dict() -> Dict[str, pd.DataFrame]:
    data = {}
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            print(f"[WARN] dataset file not found: {path}")
    return data


# -----------------------------------------------------------------------------
# main evaluation loop
# -----------------------------------------------------------------------------
def run_eval():
    datasets = load_dataset_dict()

    for model_name in MODELS:
        for ds_name, df in datasets.items():
            for temp in TEMPERATURES:
                print(f"\n=== model={model_name} | dataset={ds_name} | temp={temp} ===")

                llm = ChatOpenAI(
                    api_key=OLLAMA_API_KEY,
                    base_url=OLLAMA_BASE_URL,
                    temperature=temp,
                    model=model_name,
                )

                json_filename = f"{model_name}_{temp}_{ds_name}.json"
                metrics_filename = f"{model_name}_{temp}_{ds_name}_metrics.txt"

                if os.path.exists(json_filename):
                    with open(json_filename, "r") as f:
                        try:
                            results = json.load(f)
                        except json.JSONDecodeError:
                            results = []
                else:
                    results = []

                TP = TN = FP = FN = 0
                Total = Correct = 0
                call_times: List[float] = []

                for idx, row in df.iterrows():
                    prompt = row["Input"]
                    true_json = extract_json(row["True_output"])

                    start = time.time()
                    resp = llm.invoke(prompt)
                    elapsed = time.time() - start

                    pred_json = extract_json(resp.content)
                    call_times.append(elapsed)

                    if true_json is None:
                        # if your GT is always JSON this shouldn't happen
                        continue

                    Total += len(true_json.keys())

                    for key, true_val in true_json.items():
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

                    results.append(
                        {
                            "index": int(idx),
                            "prompt": prompt,
                            "generated_text": resp.content,
                            "true_output": true_json,
                            "time": elapsed,
                        }
                    )

                    with open(json_filename, "w") as f:
                        json.dump(results, f, indent=4)

                avg_time = sum(call_times) / len(call_times) if call_times else 0.0
                precision = TP / (TP + FP) if (TP + FP) else 0.0
                recall = TP / (TP + FN) if (TP + FN) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )

                metrics_text = (
                    f"Global Metrics:\n"
                    f"Total Correct Predictions: {Correct}/{Total}\n"
                    f"Accuracy: {Correct / Total:.2%}\n"
                    f"TP={TP}, TN={TN}, FP={FP}, FN={FN}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall:    {recall:.4f}\n"
                    f"F1 Score:  {f1:.4f}\n"
                    f"Avg Call Time: {avg_time:.4f} sec\n"
                )

                print(metrics_text)
                with open(metrics_filename, "w") as f:
                    f.write(metrics_text)


if __name__ == "__main__":
    run_eval()
