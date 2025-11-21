# Modified from https://github.com/thunlp/FR-Spec/blob/main/fr/fr.py

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os
import logging
import multiprocessing
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_data_chunk(chunk_data, model_path):
    """Process a chunk of data and return token counter"""
    # Initialize processor in each worker process
    processor = TokenizationProcessor(model_path)
    token_counter = Counter()

    # Process items in the chunk
    for item in chunk_data:
        tokens = processor.process_dataset_item(item)
        if tokens:  # Only process non-empty token lists
            token_counter.update(tokens)

    return token_counter


def merge_counters(counter_list):
    """Merge multiple Counter objects efficiently"""
    merged_counter = Counter()
    for counter in counter_list:
        merged_counter.update(counter)
    return merged_counter


class TokenizationProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_type = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"Tokenizer vocab_size: {self.tokenizer.vocab_size}")

        # Also check the model config to see the actual model vocab size
        self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model config vocab_size: {self.model_config.vocab_size}")
        print(f"Number of added special tokens: {len(self.tokenizer.added_tokens_encoder)}")

        # Handle padding token
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # Detect model type for later use with separators
        if "deepseek" in model_path.lower():
            self.model_type = "deepseek"
            logger.info(f"Detected DeepSeek model type")
            self.sep_assistant = "<｜Assistant｜>\n\n"
            self.sep_user = ""
        elif "qwen" in model_path.lower():
            self.model_type = "qwen"
            logger.info(f"Detected Qwen model type")
            self.sep_assistant = "<|im_end|>\n<|im_start|>assistant\n"
            self.sep_user = "<|im_end|>\n<|im_start|>user\n"
        elif "llama" in model_path.lower():
            self.model_type = "llama"
            logger.info(f"Detected Llama model type")
            # Llama3-style chat template
            self.sep_assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            self.sep_user = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        else:
            logger.info(f"Using generic model handling (no specific format detected)")
            # Default separators if needed
            self.sep_assistant = "<|im_end|>\n<|im_start|>assistant\n"
            self.sep_user = "<|im_end|>\n<|im_start|>user\n"

        # Calculate separator lengths
        self.sep_len_assistant = len(self.tokenizer(self.sep_assistant).input_ids)
        self.sep_len_user = len(self.tokenizer(self.sep_user).input_ids)
        logger.info(f"Separator lengths - Assistant: {self.sep_len_assistant}, User: {self.sep_len_user}")

    def process_dataset_item(self, item):
        """Process a single dataset item and return tokenized input_ids"""
        if isinstance(item, dict):
            # Handle different dataset formats
            if "conversations" in item:
                return self.process_conversation_item(item)
            elif "messages" in item:
                return self.process_messages_item(item)
            elif "text" in item:
                return self.process_text_item(item)
            elif all(field in item for field in ["prompt", "response"]):
                return self.process_prompt_response_item(item)

        # Fallback to simple text processing
        text = str(item) if not isinstance(item, dict) else str(item)
        return self.tokenizer.encode(text)

    def process_conversation_item(self, item):
        """Process a conversation item similar to EagleDatasetGenerator"""
        conversations = item["conversations"]

        if len(conversations) < 2:
            return []

        messages = []

        # Handle different conversation formats
        if isinstance(conversations[0], dict) and "value" in conversations[0]:
            # ShareGPT format
            roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

            for sentence in conversations:
                if "from" in sentence and sentence["from"] in roles:
                    role = roles[sentence["from"]]
                    content = sentence["value"]

                    # For Llama models, add a space before assistant responses
                    if self.model_type == "llama" and role == "assistant":
                        content = " " + content

                    messages.append({"role": role, "content": content})
        else:
            # Simple alternating format
            for i, content in enumerate(conversations):
                role = "user" if i % 2 == 0 else "assistant"

                text_content = ""
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, dict) and "value" in content:
                    text_content = content["value"]

                # For Llama models, add a space before assistant responses
                if self.model_type == "llama" and role == "assistant":
                    text_content = " " + text_content

                messages.append({"role": role, "content": text_content})

        return self.create_input_ids_from_messages(messages)

    def process_messages_item(self, item):
        """Process a messages item"""
        if isinstance(item["messages"], list):
            messages = []
            for msg in item["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    content = msg["content"]
                    # For Llama models, add a space before assistant responses
                    if self.model_type == "llama" and msg["role"] == "assistant":
                        content = " " + content
                    messages.append({"role": msg["role"], "content": content})

            if len(messages) > 1:
                return self.create_input_ids_from_messages(messages)

        return []

    def process_text_item(self, item):
        """Process a simple text item"""
        return self.tokenizer.encode(item["text"])

    def process_prompt_response_item(self, item):
        """Process prompt/response pair"""
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        return self.create_input_ids_from_messages(messages)

    def create_input_ids_from_messages(self, messages):
        """Create input_ids from messages using chat template"""
        if not messages:
            return []

        try:
            conversation = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            input_ids = self.tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]

            return input_ids.tolist()
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback to simple concatenation
            text = " ".join([msg["content"] for msg in messages])
            return self.tokenizer.encode(text)


def main(args):
    ds = load_from_disk(args.dataset_path)
    # ds = ds.select(range(1000))

    # Convert dataset to list for multiprocessing
    dataset_list = list(ds)

    # Setup multiprocessing
    num_processes = args.num_proc
    chunk_size = len(dataset_list) // num_processes + (len(dataset_list) % num_processes > 0)
    chunks = [dataset_list[i : i + chunk_size] for i in range(0, len(dataset_list), chunk_size)]

    logger.info(f"Processing {len(dataset_list)} samples with {num_processes} processes")
    logger.info(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

    # Process chunks in parallel
    process_func = partial(process_data_chunk, model_path=args.model_path)

    with multiprocessing.Pool(num_processes) as pool:
        logger.info("Starting parallel processing...")
        # Use tqdm to show progress for each chunk processed
        results = []
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for result in pool.imap(process_func, chunks):
                results.append(result)
                pbar.update(1)

    # Merge results
    logger.info("Merging results...")
    token_counter = merge_counters(results)

    # Calculate statistics
    num_tokens = sum(token_counter.values())
    unique_tokens = len(token_counter)

    logger.info(f"Processed {num_tokens} tokens")
    logger.info(f"Found {unique_tokens} unique tokens")

    # Initialize processor for vocab size and EOS token info
    processor = TokenizationProcessor(args.model_path)

    sort_by_freq = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 most frequent tokens:")
    for i, (token_id, freq) in enumerate(sort_by_freq[:10]):
        try:
            token_str = processor.tokenizer.decode([token_id])
            token_repr = repr(token_str) if len(token_str) > 0 else f"<token_{token_id}>"
        except:
            token_repr = f"<token_{token_id}>"
        print(f"{i+1:2d}. Token {token_id:5d} ({token_repr}): {freq} times ({freq/num_tokens*100:.2f}%)")

    ids, frequencies = zip(*sort_by_freq)
    ids = list(ids)

    print(f"\nProcessed {num_tokens} tokens from {len(dataset_list)} samples")

    if not os.path.exists(f"{args.save_path}/{args.model_name}"):
        os.makedirs(f"{args.save_path}/{args.model_name}")

    # Get vocab size for t2d mapping
    vocab_size = processor.model_config.vocab_size

    for r in args.vocab_size:
        # Use the single EOS token ID instead of encoding the string
        eos_token_id = processor.tokenizer.eos_token_id

        if eos_token_id is not None:
            if eos_token_id not in ids[:r]:
                # EOS token not in top r, add it
                freq_ids = ids[: r - 1] + [eos_token_id]
            else:
                # EOS token already in top r
                freq_ids = ids[:r]
        else:
            # No EOS token defined, just use top r
            freq_ids = ids[:r]

        print(f"EOS token ID: {eos_token_id}, included: {eos_token_id in freq_ids if eos_token_id else 'N/A'}")

        # Calculate frequency statistics for this vocab size
        top_r_frequency_sum = sum(freq for token_id, freq in sort_by_freq[:r] if token_id in freq_ids)
        top_r_ratio = top_r_frequency_sum / num_tokens
        print(f"Top {r} token frequency ratio: {top_r_ratio:.2%}")

        # Create used_tokens and sort them
        used_tokens = freq_ids.copy()
        used_tokens.sort()

        # Create d2t (draft to target) mapping
        d2t = [used_tokens[i] - i for i in range(len(used_tokens))]

        # Create t2d (target to draft) mapping
        t2d = [i in used_tokens for i in range(vocab_size)]

        # Convert to tensors
        d2t = torch.tensor(d2t)
        t2d = torch.tensor(t2d)

        # Save the mappings
        cache = {"d2t": d2t, "t2d": t2d}

        print(f"save freq_{r}.pt, size:", len(freq_ids))
        print(f"d2t tensor shape: {d2t.shape}, t2d tensor shape: {t2d.shape}")

        with open(f"{args.save_path}/{args.model_name}/freq_{r}.pt", "wb") as f:
            torch.save(cache, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5", help="The name of the model.")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="The path to the model."
    )
    parser.add_argument(
        "--dataset_path", type=str, default="dataset/eagle-mix", help="The path to the dataset."
    )
    parser.add_argument("--vocab_size", nargs="+", type=int, default=[32768], help="The vocab sizes to process.")
    parser.add_argument("--max_length", type=int, default=32768, help="Maximum sequence length for tokenization.")
    parser.add_argument("--save_path", type=str, default="freq_map", help="The path to save the output file.")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes for parallel processing.")

    args = parser.parse_args()
    # get model name from model_path
    model_name = os.path.basename(args.model_path)
    args.model_name = model_name

    print(args)
    main(args)
