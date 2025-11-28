import time
import pickle
import random
import argparse
import os
import datetime
from tqdm import tqdm

from utils import load_cladder, prepare_prompt, save_batch, update_final_acc
from generation import ollama_gen, summarize, debate_turn

random.seed(42)

MODELS = {"qwen": "qwen3:32b", "deepseek": "deepseek-r1:32b"}
MODEL_NAMES = list(MODELS.keys())
SUMMARIZER = "granite3.3:2b"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

    parser.add_argument("--ranges", default="10000-10111", type=str)
    parser.add_argument("--ckpt_freq", default=200, type=int)
    parser.add_argument("--summary_freq", default=500, type=int)
    parser.add_argument("--hardware", default="alien", type=str)
    parser.add_argument("--responses", default=4, type=int)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--output_file", default=f"results_{timestamp}.pkl", type=str)

    args = parser.parse_args()

    full_t = time.time()
    data = load_cladder()
    start_idx = 0

    # Parse sample ranges to run through
    idx_range = []
    for r in args.ranges.split(","):
        start, end = map(int, r.split("-"))
        idx_range.append((start, end + 1))

    # Collect relevant samples
    test_samples = []
    for start, end in idx_range:
        subset = data.to_dict(orient="records")[start:end]
        test_samples.extend(subset)

    print(f"Total number of samples: {len(test_samples)}")

    # Initialize trackers
    all_results = []
    init_acc = {name: {"correct": 0, "total": 0} for name in MODEL_NAMES}
    final_acc = {name: {"correct": 0, "total": 0} for name in MODEL_NAMES}
    gen_times = {name: [] for name in MODEL_NAMES}

    # Resume from checkpoint if provided
    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f"--- Loading state from checkpoint: {args.load_ckpt} ---")
        with open(args.load_ckpt, "rb") as f:
            ckpt = pickle.load(f)

        start_idx = ckpt.get("last_idx", -1) + 1
        init_acc = ckpt["init_acc"]
        final_acc = ckpt["final_acc"]
        gen_times = ckpt["gen_times"]
        print(f"Resuming from sample index {start_idx}")

    # Main loop
    for i in tqdm(
        range(start_idx, len(test_samples)), initial=start_idx, total=len(test_samples)
    ):
        s = {}
        print(f"\n\n----- Processing Sample {i} -----")
        samp = test_samples[i]

        # Copy relevant fields to output sample
        s.update(
            {
                "gold_answer": samp["answer"],
                "initial_question": samp["question"],
                "question_id": samp["question_id"],
                "rung": samp["rung"],
                "difficulty": samp["difficulty"],
            }
        )

        # Flip a coin for first speaker
        first_name = random.choice(MODEL_NAMES)
        second_name = [name for name in MODEL_NAMES if name != first_name][0]
        s["first_speaker"] = first_name
        print(f"Initial speaker for this sample: {first_name}")

        # First speaker generates response
        print(f"--- Round 0: Initial Response by {first_name} ---")
        prompt = prepare_prompt(samp, 0)
        raw_result, gen_time = ollama_gen(MODELS[first_name], prompt, format=None)

        # Save results
        print(f"Time for initial generation (Sample {i}): {gen_time:.2f} seconds")
        s["timing_0"] = gen_time
        s["raw_output_0"] = (raw_result, first_name)
        gen_times[first_name].append(gen_time)

        # Extract final yes/no answer and confidence estimation
        start_t_summ = time.time()
        summary = summarize(SUMMARIZER, raw_result, first_name)
        summ_t = time.time() - start_t_summ

        # Save results
        print(f"  Time to summarize initial round: {summ_t:.2f} seconds")
        s["timing_summ"] = summ_t
        s["output_0"] = summary

        # Track initial accuracy
        init_speak_samp = s["first_speaker"]
        init_acc[init_speak_samp]["total"] += 1
        if (
            str(s["gold_answer"]).strip().lower()
            == str(summary.get("answer", "")).strip().lower()
        ):
            init_acc[init_speak_samp]["correct"] += 1

        speaker_order = {
            r_idx: (second_name if r_idx % 2 != 0 else first_name)
            for r_idx in range(1, args.responses)
        }

        # Debate loop
        for r in range(1, args.responses):
            curr_name, opp_name = (
                speaker_order[r],
                first_name if speaker_order[r] == second_name else second_name,
            )
            print(
                f"\n--- Round {r}: {curr_name} debates {opp_name}'s output from Round {r - 1} ---"
            )

            # Generate debate response
            s = debate_turn(
                curr_name, MODELS[curr_name], samp, s, r, opp_name, SUMMARIZER
            )
            gen_times[curr_name].append(s[f"timing_{r}"])
            # If both models agree on the answer, stop debate early
            if s[f"output_{r}"].get("answer") == s[f"output_{r - 1}"].get("answer"):
                break

        all_results.append(s)

        # Checkpointing
        if (i + 1) % args.ckpt_freq == 0:
            ckpt = {
                "last_idx": i,
                "init_acc": init_acc,
                "final_acc": final_acc,
                "gen_times": gen_times,
            }
            with open(f"ckpt_{args.hardware}", "wb") as f:
                pickle.dump(ckpt, f)
            print(
                f"\n[Checkpoint saved] Progress for trackers saved to checkpoint_{args.hardware}"
            )

        # Periodic summary and batch save logic
        if (i + 1) % args.summary_freq == 0 and i > start_idx:
            print(
                f"\n--- Reached summary interval of {args.summary_freq}. Processing and saving batch... ---"
            )

            final_acc = update_final_acc(all_results, final_acc)
            save_batch(args.output_file, all_results)

            # List can grow quite large, so periodically save and reset it
            print("--- Resetting in-memory results list. ---")
            all_results = []

    print("\n--- End of Run: Processing and Saving Final Batch ---")
    if all_results:
        final_acc = update_final_acc(all_results, final_acc)
        save_batch(args.output_file, all_results)

    # Save the final cumulative summary text file
    full_time_end = time.time() - full_t

    print(f"\nRun complete. All results are saved in {args.output_file}")
    print(
        f"Time taken: {full_time_end / 60:.2f} minutes ({full_time_end / 60 / 60:.2f} hours)"
    )
