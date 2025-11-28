import os
import pickle
import re
import ast
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from pydantic import BaseModel

random.seed(42)


def timeit(func):
    """Decorator to time a function's execution."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    return wrapper


def load_cladder():
    """Load all samples of the CLadder 'balanced' variant, with all relevant fields."""
    start_t = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(
        script_dir, "..", "data", "cladder-v1", "cladder-v1-q-balanced.json"
    )
    meta_path = os.path.join(script_dir, "..", "data", "cladder-v1-meta-models.json")

    df = pd.read_json(df_path)
    meta = pd.read_json(meta_path)

    df["rung"] = df["meta"].apply(lambda x: x.get("rung"))
    df["model_id"] = df["meta"].apply(lambda x: x.get("model_id"))

    # Fetch background and difficulty from meta file
    model_backgrounds = {
        m["model_id"]: m["background"] for m in meta.to_dict(orient="records")
    }
    model_difficulty = {
        m["model_id"]: m["difficulty"] for m in meta.to_dict(orient="records")
    }
    df["background"] = df["model_id"].map(model_backgrounds)
    df["difficulty"] = df["model_id"].map(model_difficulty)

    print(f"Time taken to load data: {time.time() - start_t:.2f} seconds")
    return df.drop(columns=["meta"])


def invalid_result():
    """Returns a default invalid result dictionary."""
    result = {
        "reasoning": "",
        "answer": np.random.choice(["yes", "no"]),
        "confidence": 0.0,
        "model": "error",
    }
    return result


def prepare_prompt(test_sample, round_num, history=None):
    """Craft the elaborate prompt needed for causal debate, personalized for debate round and speaker role."""
    prompt = []

    if round_num == 0:
        # The first portion of the prompt is the same as CausalCoT
        prompt.append("""
You will be asked a causal reasoning question. You should structure your final answer as follows: 
Step 1) Extract the causal graph: Identify the causal graph that depicts the relationships in the scenario. 
The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.  
Step 2) Determine the query type: Identify the type of query implied by the main question. Choices 
include "marginal probability", "conditional probability", "explaining away effect", "backdoor adjustment set", 
"average treatment effect", "collider bias", "normal counterfactual question", "average treatment effect on treated", 
"natural direct effect" or "natural indirect effect". Your answer should only be a term from the list above, 
enclosed in quotation marks.  
Step 3) Formalize the query: Translate the query into its formal mathematical 
expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.  
Step 4) Gather all relevant data: Extract all the available data. Your answer should contain nothing but 
marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each 
probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.  
Step 5) Deduce the estimand using causal inference: Given all the information above, deduce the estimand using 
skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.  
Step 6) Calculate the estimand: Insert the relevant data in Step 4 into the estimand, perform basic 
arithmetic calculations, and derive the final answer. 
Step 7) Give a final yes/no answer to the question.
    """)
    else:
        # I only mention the debate format after the first round because I don't want to distract the first speaker
        prompt.append("""
You and another LLM are being asked a causal reasoning question. I believe two heads are better than one, so I ask that you 
debate with the other LLM to reach a consensus on the answer. You will be shown the query, then the current history of your 
debate will be provided for reference.
    """)

    # Provide all details of the question
    prompt.append(
        f"{test_sample['background']}\n\n{test_sample['given_info']}\n\n**Question**: {test_sample['question']}\n"
    )
    # Without this, models were more likely to doubt their answers repeatedly, leading to much longer generation times
    prompt.append("""
There is an identifiable yes/no answer, which may sometimes go against your commonsense intuition. Be confident in your 
thinking: while answers may be unintuitive, there are no trick questions, and answers will be obvious once calculated.
""")

    if history:
        prompt.append("\n".join(history))

        # Tweak instructions based on speaker role
        if round_num % 2 != 0:
            # Odd rounds (1, 3, ...) are second speaker's turn
            prompt.append("""
Carefully read the causal query, then scrutinize your opponent's solution. If you identify any flaws in their 
reasoning or errors in their calculations, you should point them out and suggest corrections. You should make 
explicit references to your opponent's response.
""")
        else:
            # Even rounds (2, 4, ...) are first speaker's turn
            prompt.append("""
You should carefully consider your opponent's response, and then may defend your previous answers, or be 
persuaded to your opponent's solution, as you see fit. You should make explicit references to your opponent's response.
""")

    prompt.append("""
After discussing your rationale, it is crucial that you give a final yes/no answer to the causal query. Do not answer with 
a number: answer yes or no only. You should also explicitly state your level of confidence in your answer (between 0.0 and 1.0).\n
""")
    print("\n".join(prompt))
    return "\n".join(prompt)


def update_final_acc(results_batch, tracker):
    """
    Iterates through a batch of results and updates the final accuracy tracker.
    """
    for res in results_batch:
        # Find the last valid output from the debate rounds
        final_out_key = next(
            (f"output_{r}" for r in range(len(res), -1, -1) if f"output_{r}" in res),
            None,
        )
        if (
            res.get("first_speaker")
            and final_out_key
            and res[final_out_key] is not None
        ):
            speaker = res["first_speaker"]
            tracker[speaker]["total"] += 1
            if (
                str(res["gold_answer"]).strip().lower()
                == str(res[final_out_key].get("answer", "")).strip().lower()
            ):
                tracker[speaker]["correct"] += 1
    return tracker


def save_batch(output_file, results_batch):
    """
    Saves a batch of results to a cumulative pickle file.
    """
    if not results_batch:
        print("No new results in this batch to save.")
        return

    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                existing_results = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            print(f"Warning: Could not read {output_file}. A new file will be created.")
            existing_results = []
    else:
        existing_results = []

    existing_results.extend(results_batch)
    with open(output_file, "wb") as f:
        pickle.dump(existing_results, f)

    print(
        f"Saved {len(results_batch)} new results. Total results in {output_file}: {len(existing_results)}"
    )
