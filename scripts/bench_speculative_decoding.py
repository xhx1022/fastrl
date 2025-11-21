import argparse
import json
import os
import time

import sglang as sgl
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def load_json_file(json_file):
    with open(json_file, "r") as f:
        prompts = json.load(f)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-r1", help="Path to the model")
    parser.add_argument("--data_dir", type=str, default="Eurus-Data", help="Path to the data directory")
    parser.add_argument("--spec_algorithm", type=str, default=None, help="Speculative algorithm to use")
    parser.add_argument("--eagle_path", type=str, default=None, help="Path to the eagle model")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--speculative_num_steps", type=int, default=8, help="Number of speculative steps")
    parser.add_argument("--speculative_eagle_topk", type=int, default=8, help="Top k for eagle")
    parser.add_argument("--speculative_num_draft_tokens", type=int, default=64, help="Number of draft tokens")
    parser.add_argument("--max_bs", type=int, default=8, help="Max batch size")
    parser.add_argument("--attention_backend", type=str, default="fa3", help="Attention backend to use")

    args = parser.parse_args()

    spec_algorithm = args.spec_algorithm

    if spec_algorithm == "None":
        spec_algorithm = None
    
    model_path = args.model_path
    processed_data_dir = args.data_dir

    prompts = load_json_file(processed_data_dir)

    # Create a sampling params object.
    sampling_params = {
        "n": 1,
        "temperature": 0.6,
        "max_new_tokens": 2048,
    }
    # Set speculative args based on algorithm
    speculative_args = {}
    if spec_algorithm == "EAGLE3":
        speculative_args = {
            "speculative_algorithm": "EAGLE3",
            "speculative_draft_model_path": args.eagle_path,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
        }
    elif spec_algorithm == "EAGLE":
        speculative_args = {
            "speculative_algorithm": "EAGLE",
            "speculative_draft_model_path": args.eagle_path,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
        }
    elif spec_algorithm == "LOOKAHEAD":
        speculative_args = {
            "speculative_algorithm": "LOOKAHEAD",
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
            "speculative_lookahead_one_branch": True,
        }

    # Create an LLM.
    llm = sgl.Engine(
        model_path=model_path,
        dtype="bfloat16",
        cuda_graph_max_bs=args.max_bs,
        tp_size=args.tp_size,
        max_running_requests=args.max_bs,
        mem_fraction_static=0.6,
        context_length=4096,
        attention_backend=args.attention_backend,
        **speculative_args,
    )

    total_runs = 2

    total_prompts = len(prompts)
    if total_prompts < args.max_bs:
        # Repeat prompts to meet minimum batch size
        prompts = prompts * (args.max_bs // total_prompts)
        prompts = prompts[:args.max_bs]

    for idx in range(total_runs):
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        cos = time.time() - start
        completion_tokens = 0
        verify_tokens = 0

        # Print the outputs.
        for output in outputs:
            completion_tokens += output["meta_info"]["completion_tokens"]
            has_verify = "spec_verify_ct" in output["meta_info"]
            if has_verify:
                verify_tokens += output["meta_info"]["spec_verify_ct"]
            else:
                verify_tokens += output["meta_info"]["completion_tokens"]

        print("======================" * 3)
        print("Tested Reqs Num: ", len(outputs))
        print(f"completion_tokens {completion_tokens}, verify_tokens {verify_tokens}, {cos:.4f}s")
        accept_length = completion_tokens / verify_tokens if verify_tokens > 0 else 1.0
        print(f"Run:{idx + 1}, {spec_algorithm}, Accept length: {accept_length:.3f}, TPS: {completion_tokens/cos:.2f}\n")
        
        if idx == total_runs - 1:
            # Write to csv file
            with open(f"speculative_{args.spec_algorithm}_stats.csv", "a") as f:
                f.write(f"{args.max_bs},{args.speculative_num_steps},{args.speculative_eagle_topk},{args.speculative_num_draft_tokens},{accept_length:.3f},{completion_tokens/cos:.2f}\n")


if __name__ == "__main__":
    main()
