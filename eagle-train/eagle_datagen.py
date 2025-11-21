import os
import json
import logging
import torch
import torch.distributed as dist
import hydra
import pandas as pd
from pathlib import Path
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.device_mesh import init_device_mesh
from utils import initialize_global_process_group

import glob

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "INFO"))


class EagleDatasetGenerator:

    def __init__(self, datapath, max_len=32768, base_model_path=None, save_dir=None, sample_ratio=1.0, mode="eagle2"):
        self.max_len = max_len
        self.base_model_path = base_model_path
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_type = None
        self.sample_ratio = sample_ratio
        self.mode = mode.lower()
        if self.mode not in ["eagle2", "eagle3"]:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes are 'eagle2' and 'eagle3'.")
        self.processed_dataset = []
        # Load tokenizer
        if base_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

            # Handle padding token
            if not self.tokenizer.pad_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

            # Detect model type for later use with separators
            if "deepseek" in base_model_path.lower():
                self.model_type = "deepseek"
                logger.info(f"Detected DeepSeek model type")
                self.sep_assistant = "<｜Assistant｜>\n\n"
                self.sep_user = "<｜User｜>"
            elif "qwen" in base_model_path.lower():
                self.model_type = "qwen"
                logger.info(f"Detected Qwen model type")
                self.sep_assistant = "<|im_end|>\n<|im_start|>assistant\n"
                self.sep_user = "<|im_end|>\n<|im_start|>user\n"
            elif "llama" in base_model_path.lower():
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

        # Load the data using datasets library
        self.load_data(datapath)

        # Distribute data to different workers
        self.distribute_data()

        # Process the dataset
        self.process_dataset(self.worker_data)

        # Load base model for getting hidden states
        if base_model_path:
            # Initialize base model
            logger.info(f"Rank {self.rank}/{self.world_size}: Loading base model...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,  # torch.bfloat16,
                device_map={"": f"cuda:{self.local_rank}"},
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

            self.base_model.eval()
            
            # Enable gradient checkpointing to save memory
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()

            # Get model's maximum context length
            self.model_max_length = getattr(self.base_model.config, "max_position_embeddings", self.max_len)
            logger.info(f"Model's maximum context length: {self.model_max_length}")
            # Update max_len to be within model's limits
            self.max_len = min(self.max_len, self.model_max_length)
            logger.info(f"Using maximum sequence length: {self.max_len}")

            self.worker_data = self.processed_dataset

    def load_data(self, datapath):
        """Load data using datasets library and process conversations"""
        self.dataset = []

        # Try to load the dataset with appropriate format detection
        logger.info(f"Loading dataset from {datapath}")
        parquet_files = glob.glob(os.path.join(datapath, "data", "*.parquet"))
        if parquet_files:
            full_dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        else:
            full_dataset = load_from_disk(datapath)

        # Calculate number of samples
        ratio = self.sample_ratio
        num_samples = len(full_dataset)
        samples_to_keep = int(num_samples * ratio)

        indices = torch.randperm(num_samples)[:samples_to_keep].tolist()

        # Select the sampled data
        self.dataset = full_dataset.select(indices)
        logger.info(f"Loaded {len(self.dataset)} samples ({samples_to_keep}/{num_samples}, {ratio*100}% of total data)")
        del full_dataset

    def process_dataset(self, ds):
        """Process items from a HuggingFace dataset or list of items"""
        if hasattr(ds, "column_names"):
            # Handle HuggingFace dataset
            if "conversations" in ds.column_names:
                # Process conversations field
                for item in tqdm(ds, desc=f"[Rank{self.rank}] Processing conversations", position=self.rank):
                    self.process_conversation_item(item)
            elif "messages" in ds.column_names:
                # Process messages field
                for item in tqdm(ds, desc=f"[Rank{self.rank}] Processing messages", position=self.rank):
                    if isinstance(item["messages"], list):
                        messages = []
                        for msg in item["messages"]:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                messages.append({"role": msg["role"], "content": msg["content"]})

                        if len(messages) > 1:
                            self.create_conversation_entry(messages)
            elif all(field in ds.column_names for field in ["prompt", "response"]):
                # Handle simple prompt/response pairs
                for item in tqdm(ds, desc=f"[Rank{self.rank}] Processing prompt/response pairs", position=self.rank):
                    self.format_conversation(item["prompt"], item["response"])
        else:
            # Handle list of items
            for item in tqdm(ds, desc=f"[Rank{self.rank}] Processing list of items", position=self.rank):
                if isinstance(item, dict):
                    if "conversations" in item:
                        self.process_conversation_item(item)
                    elif "messages" in item:
                        if isinstance(item["messages"], list):
                            messages = []
                            for msg in item["messages"]:
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    messages.append({"role": msg["role"], "content": msg["content"]})
                            if len(messages) > 1:
                                self.create_conversation_entry(messages)
                    elif all(field in item for field in ["prompt", "response"]):
                        self.format_conversation(item["prompt"], item["response"])
                    else:
                        # Try to infer the format from the item
                        for key, value in item.items():
                            if isinstance(value, list) and len(value) >= 2:
                                # Might be a conversation
                                self.process_conversation_item({key: value})

    def process_conversation_item(self, item):
        """Process a single conversation item"""
        if hasattr(item, "conversations") or "conversations" in item:
            conversations = item.conversations if hasattr(item, "conversations") else item["conversations"]

            if len(conversations) >= 2:
                messages = []

                # Handle different conversation formats
                if isinstance(conversations[0], dict) and "value" in conversations[0]:
                    # ShareGPT format
                    roles = {"human": "user", "gpt": "assistant"}
                    source = conversations

                    # # Skip if first message is not from human
                    # if "from" in source[0] and roles.get(source[0]["from"]) != "user":
                    #     source = source[1:]

                    for j, sentence in enumerate(source):
                        # if "from" in sentence and sentence["from"] in roles:
                        role = sentence["from"]
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

                # Only process if we have valid messages
                if len(messages) > 1:
                    self.create_conversation_entry(messages)

    def create_conversation_entry(self, messages):
        """Create conversation entry from messages"""
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            logger.warning("Tokenizer not available for processing")
            return None

        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # return_dict=True,
            # return_tensors="pt",
            # add_generation_prompt=True,
            # return_assistant_tokens_mask=True,
        )

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        input_ids = self.tokenizer(
            conversation,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        ).input_ids[0]

        loss_mask = torch.ones_like(input_ids)

        # Unified implementation for handling different model types
        if self.model_type in ["qwen", "llama", "deepseek"]:
            turns = conversation.split(self.sep_user)

            if len(turns) > 1:
                # First turn contains the system message, combine it with the first user message
                turns[1] = turns[0] + self.sep_user + turns[1]
                turns = turns[1:]  # Remove the system message turn

                cur_len = 1
                loss_mask[:cur_len] = 0  # Mask out the beginning token

                for i, turn in enumerate(turns):
                    if turn == "":
                        break

                    turn_len = len(self.tokenizer(turn).input_ids)

                    parts = turn.split(self.sep_assistant)
                    if len(parts) != 2:
                        break

                    parts[0] += self.sep_assistant
                    instruction_len = len(self.tokenizer(parts[0]).input_ids)

                    # Adjust the loss mask based on model type
                    if self.model_type == "qwen" or self.model_type == "deepseek":
                        if i == 0:
                            # For the first turn, mask out the user part
                            loss_mask[0 : cur_len + instruction_len - 2] = 0
                        else:
                            # For subsequent turns, mask out the user part
                            loss_mask[cur_len - 2 : cur_len + instruction_len - 1] = 0

                        cur_len += turn_len
                        cur_len += 2  # Adjust for separator
                    elif self.model_type == "llama":
                        if i == 0:
                            loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0

                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3

                # Mask out the rest of the sequence
                loss_mask[cur_len:] = 0
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.processed_dataset.append({"input_ids": input_ids, "loss_mask": loss_mask, "conversation": conversation})

        return True

    def format_conversation(self, prompt, response):
        """Format conversation from prompt/response pair"""
        if not prompt or not response:
            return None

        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        return self.create_conversation_entry(messages)

    def distribute_data(self):
        """Distribute data to different workers"""
        num_samples = len(self.dataset)
        worker_indices = []

        for i in range(self.rank, num_samples, self.world_size):
            worker_indices.append(i)

        self.worker_indices = worker_indices
        self.worker_data = [self.dataset[i] for i in worker_indices]

        logger.info(f"Rank {self.rank}/{self.world_size} has {len(self.worker_indices)} samples")
        del self.dataset

    @torch.no_grad()
    def process_data(self):
        """Process data and generate hidden states"""
        dropped_count = 0
        max_seq_len = 0

        # Create progress bar for this rank
        desc = f"Rank {self.rank}/{self.world_size}"
        pbar = tqdm(enumerate(self.worker_data), total=len(self.worker_data), desc=desc, position=self.rank)

        for i, data_point in pbar:
            orig_idx = self.worker_indices[i]
            
            output_path = os.path.join(self.save_dir, f"data_{orig_idx}.pt")
            if os.path.exists(output_path):
                print(f"Skipping data_{orig_idx} because it already exists")
                continue
            
            input_ids = data_point["input_ids"]
            seq_len = len(input_ids)

            # Skip sequences longer than 32K
            if seq_len > 32768:
                dropped_count += 1
                continue

            # Process the sequence
            input_ids = input_ids.unsqueeze(0).to(self.get_device())
            loss_mask = data_point["loss_mask"]

            max_seq_len = max(max_seq_len, seq_len)

            if self.mode == "eagle2":
                # Extract hidden states
                with torch.no_grad():
                    outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
                    num_layers = self.base_model.config.num_hidden_layers
                    # Eagle2: [num_layers]
                    hidden_state = outputs.hidden_states[-1]

                # Store processed data
                processed_data = {
                    "index": orig_idx,
                    "input_ids": input_ids.cpu().squeeze(0),
                    "hidden_state": hidden_state.cpu().squeeze(0),
                    "loss_mask": loss_mask,
                    "seq_len": seq_len,
                }
            elif self.mode == "eagle3":
                # Extract hidden states
                with torch.no_grad():
                    outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
                    num_layers = self.base_model.config.num_hidden_layers
                    # Eagle3: [2, num_layers // 2, num_layers - 3]
                    target_layers = [2, num_layers // 2, num_layers - 3, num_layers]
                    hidden_states_dict = {f"layer_{i}": outputs.hidden_states[i].bfloat16().cpu() for i in target_layers}
                    hidden_states = torch.cat(list(hidden_states_dict.values()), dim=-1)
                    # hidden_state = outputs.hidden_states[-1]

                # Store processed data
                processed_data = {
                    "index": orig_idx,
                    "input_ids": input_ids.cpu().squeeze(0),
                    "hidden_states_dict": hidden_states_dict,
                    "hidden_states": hidden_states.squeeze(0),
                    # "hidden_state": hidden_state.cpu().squeeze(0),
                    "loss_mask": loss_mask,
                    "seq_len": seq_len,
                }

            # Save directly to save_dir
            # output_path = os.path.join(self.save_dir, f"data_{orig_idx}.pt")
            torch.save(processed_data, output_path)

            # Update progress bar postfix with sequence length info
            pbar.set_postfix(seq_len=seq_len)

        # Save statistics (only rank 0)
        if self.rank == 0:
            stats = {"max_seq_len": max_seq_len, "dropped_count": dropped_count}
            stats_file = os.path.join(self.save_dir, "stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved statistics to {stats_file}")
            logger.info(f"Dropped sequences: {dropped_count}")

    def get_device(self):
        """Get the device for the current process"""
        if hasattr(self.base_model, "device"):
            return self.base_model.device
        else:
            return f"cuda:{self.local_rank}"


@hydra.main(config_path="config", config_name="datagen_config", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.barrier()

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))

    generator = EagleDatasetGenerator(
        datapath=config.data.data_path,
        max_len=config.data.max_length,
        base_model_path=config.model.base_model_path,
        save_dir=config.data.save_dir,
        sample_ratio=config.data.sample_ratio,
        mode=config.data.mode,
    )
    generator.process_data()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()  # Hydra automatically provides the config parameter
