import numpy as np
import os
import time
import random
from tqdm import tqdm
import requests
from pydantic import BaseModel
import ast

from utils import invalid_result, prepare_prompt, timeit

random.seed(42)

OLLAMA_URL = "http://localhost:11434/api/generate"


# Structured output for extractor model
class Answer(BaseModel):
    answer: str
    confidence: float


@timeit
def ollama_gen(model, prompt, ctx_size=2048, format=None):
    """Generates a response from the Ollama API and times the request."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.6, "top_p": 0.95, "num_ctx": ctx_size},
    }

    # Add format to payload only if provided
    if format:
        payload["format"] = format
        payload["think"] = False

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    content = response.json()["response"]

    return content.strip()


def summarize(summ_id, raw_text, model_name):
    """Extract a yes/no answer and estimation of confidence from a reasoning model's final output."""

    # Exclude thinking trace in raw output
    final = raw_text[raw_text.find("</think>") :]

    prompt = f"""
    The following is a response from another AI model. Your task is to extract the final 'answer' (which should be 'yes' or 'no'), and 'confidence' (a float between 0.0 and 1.0) from the text below. 
    If confidence is not specified, assume it is 0.6.

    Response:
    ---
    {final}
    ---

    Output with JSON format as follows: {{"answer": "", "confidence": ""}}. Only answer yes or no in the "answer" field.
    """

    summ_response, _ = ollama_gen(
        summ_id,
        prompt,
        format=Answer.model_json_schema(),  # Enforce format structure
    )
    try:
        parsed_json = ast.literal_eval(summ_response)
        parsed_json["model"] = model_name
        parsed_json["reasoning"] = final

        if isinstance(parsed_json, dict):
            # Extract confidence
            try:
                parsed_json["confidence"] = float(parsed_json.get("confidence", 0.0))
            except (ValueError, TypeError):
                parsed_json["confidence"] = 0.0

            # Extract yes/no answer
            answer = str(parsed_json.get("answer", "")).strip().lower()
            if answer not in ["yes", "no"]:
                parsed_json["answer"] = np.random.choice(["yes", "no"])
            else:
                parsed_json["answer"] = answer

            return parsed_json
        else:
            print(f"Summarizer failed to parse JSON: {parsed_json}")
            return invalid_result()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return invalid_result()


def debate_turn(
    curr_name,
    curr_id,
    test_sample,
    samp_res,
    round_num,
    opp_name,
    summ_id,
):
    """Handles a single turn of the debate."""

    history = []
    first_speaker = samp_res.get("first_speaker")

    for j in range(0, round_num):
        key = samp_res[f"output_{j}"]

        # Determine current speaker's role
        curr_speaker = ""
        if j == 0:
            curr_speaker = first_speaker
        else:
            curr_speaker = opp_name if key.get("model") == opp_name else curr_name

        # Clearly delineate which response belongs to who
        if curr_speaker == opp_name:
            history.append("Opponent's response:")
        else:
            history.append("Your response:" if j > 0 else "Your original answer:")
        history.append(f"\n==========\n\n{key.get('reasoning')}\n\n==========\n")

    prompt = prepare_prompt(test_sample, round_num, history)

    # Increase context size as rounds progress to accomodate growing chat history
    ctx_size = 2048 * (round_num + 1)

    result, gen_time = ollama_gen(
        curr_id,
        prompt,
        ctx_size=ctx_size,
        format=None,
    )
    # Save results
    print(
        f"  Time taken for answer generation (Debate Round {round_num}): {gen_time:.2f} seconds"
    )
    samp_res[f"raw_output_{round_num}"] = (result, curr_name)
    samp_res[f"timing_{round_num}"] = gen_time

    # Extract final yes/no answer and confidence estimation
    start_t_summ = time.time()
    summary = summarize(summ_id, result, curr_name)
    samp_res[f"output_{round_num}"] = summary
    end_t_summ = time.time()

    # Save timing result
    summ_t = end_t_summ - start_t_summ
    print(f"  Time taken to summarize (Debate Round {round_num}): {summ_t:.2f} seconds")
    samp_res["timing_summ"] += summ_t

    return samp_res
