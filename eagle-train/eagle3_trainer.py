import os
import logging
import json
import gc
import deepspeed
import argparse

from safetensors.torch import safe_open
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.llama_eagle3 import LlamaModelEagle3
from model.qwen2_eagle3 import Qwen2ModelEagle3
from model.qwen3_eagle3 import Qwen3ModelEagle3
from utils import Tracking

set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def log_memory_stats(rank, prefix=""):
    """Log GPU memory statistics for debugging"""
    if torch.cuda.is_available() and rank == 0:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        logger.info(f"{prefix} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


def add_args():
    parser = argparse.ArgumentParser(description="Eagle3 Trainer DeepSpeed")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument(
        "--freq_map_path", type=str, default="freq_map/llama3/freq_32768.pt", help="Path to frequency mapping file"
    )
    parser.add_argument("--draft_vocab_size", type=int, default=32768, help="Draft vocabulary size")
    parser.add_argument("--prediction_length", type=int, default=7, help="Number of prediction steps")
    parser.add_argument("--project_name", type=str, default="Eagle3-Trainer", help="WandB project name")
    parser.add_argument("--experiment_name", type=str, default="eagle3-deepspeed", help="WandB experiment name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--load_optimizer", action="store_true", help="Whether to load optimizer states")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Training precision type")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps. If None, save every epoch.")

    # Online target model inference support
    parser.add_argument("--use_target_model", action="store_true", help="Use target model to generate hidden states")
    parser.add_argument(
        "--target_model_layers", type=str, default="0,1,2", help="Comma-separated list of layer indices to extract"
    )

    parser = deepspeed.add_config_arguments(parser)
    return parser


class EagleDataset(Dataset):

    def __init__(
        self,
        datapath,
        max_len=2048,
        dataset_max_len=None,
        target_model=None,
        tokenizer=None,
        use_target_model=False,
        target_model_layers=[0, 1, 2],
    ):
        """Initialize EagleDataset to load pre-processed data or generate hidden states dynamically"""
        self.datapath = datapath
        self.max_len = max_len
        self.global_max_seq_len = dataset_max_len
        self.failed_indices = set()  # Track failed loads

        # Target model support
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.use_target_model = use_target_model
        self.target_model_layers = target_model_layers

        if self.use_target_model and (self.target_model is None or self.tokenizer is None):
            raise ValueError("target_model and tokenizer must be provided when use_target_model=True")

    def __len__(self):
        return len(self.datapath)

    @torch.no_grad()
    def _generate_hidden_states(self, input_ids, attention_mask):
        """Generate hidden states using target model"""
        if not self.use_target_model:
            return None

        # Ensure tensors are on the same device as target model
        device = next(self.target_model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate outputs with hidden states
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
            )

        # Extract hidden states from specified layers
        hidden_states_list = []
        for layer_idx in self.target_model_layers:
            if layer_idx < len(outputs.hidden_states):
                hidden_states_list.append(outputs.hidden_states[layer_idx])

        # Concatenate hidden states from different layers
        if len(hidden_states_list) > 1:
            hidden_states = torch.cat(hidden_states_list, dim=-1)
        else:
            hidden_states = hidden_states_list[0]

        return hidden_states.squeeze(0)  # Remove batch dimension

    def __getitem__(self, idx):
        if idx in self.failed_indices:
            return self.__getitem__((idx + 1) % len(self.datapath))

        if self.use_target_model:
            # Load raw text data for dynamic hidden state generation
            if self.datapath[idx].endswith(".json"):
                with open(self.datapath[idx], "r") as f:
                    data = json.load(f)

                # Expect format: {"text": "...", "input_ids": [...], "attention_mask": [...]}
                if "input_ids" in data:
                    input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
                    attention_mask = torch.tensor(data.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
                else:
                    # Tokenize text if only text is provided
                    text = data.get("text", "")
                    encoded = self.tokenizer(text, return_tensors="pt", max_length=self.max_len, truncation=True)
                    input_ids = encoded["input_ids"].squeeze(0)
                    attention_mask = encoded["attention_mask"].squeeze(0)

                # Generate loss mask (assuming all tokens contribute to loss for now)
                loss_mask = torch.ones_like(input_ids, dtype=torch.float)

                # Generate hidden states using target model
                hidden_states = self._generate_hidden_states(input_ids, attention_mask)

            else:
                # For .pt files, assume they contain input_ids and possibly other data
                data = torch.load(self.datapath[idx], weights_only=True)
                input_ids = data["input_ids"]
                attention_mask = data.get("attention_mask", torch.ones_like(input_ids))
                loss_mask = data.get("loss_mask", torch.ones_like(input_ids, dtype=torch.float))

                # Generate hidden states using target model
                hidden_states = self._generate_hidden_states(input_ids, attention_mask)

            processed_item = {
                "input_ids": input_ids[1:],  # Shift right by 1
                "hidden_states": hidden_states[:-1] if len(hidden_states) > 1 else hidden_states,  # Current hidden states
                "loss_mask": loss_mask[1:] if len(loss_mask) > 1 else loss_mask,  # Shift loss mask
                "max_seq_len": self.global_max_seq_len,
            }
        else:
            try:
                data = torch.load(self.datapath[idx], weights_only=True)
            except Exception as e:
                return self.__getitem__((idx + 1) % len(self.datapath))

            # Skip samples that are longer than max_len
            if len(data["input_ids"]) > self.max_len:
                self.failed_indices.add(idx)
                return self.__getitem__((idx + 1) % len(self.datapath))

            data["loss_mask"][-1] = 0
            all_hidden_states = list(data["hidden_states_dict"].values())
            hidden_states = torch.cat(all_hidden_states[:-1], dim=-1).squeeze(0)
            last_hidden_states = all_hidden_states[-1].squeeze(0)
            processed_item = {
                "input_ids": data["input_ids"][1:],
                "hidden_states": hidden_states,
                "last_hidden_states": last_hidden_states,
                "loss_mask": data["loss_mask"],
                "max_seq_len": self.global_max_seq_len,
            }

        return processed_item


class EagleDataCollator:

    def padding_tensor(self, input, N):
        input = input.unsqueeze(0)

        if input.dim() < 4:
            B, n = input.shape[:2]
        else:
            assert input.shape[1] == 3, f"got {input.shape}"  # Eagle-3 uses 3 layers' hidden_states [1, 3, n, d]
            B, _, n = input.shape[:3]

        padding_length = N - n
        if len(input.shape) == 2:  # [B, n]
            output = torch.nn.functional.pad(input, (0, padding_length), value=0)
        elif len(input.shape) >= 3:  # [B, n, d] / [B, 3, n, d] for hidden states
            output = torch.nn.functional.pad(input, (0, 0, 0, padding_length), value=0)
        else:
            raise ValueError(f"Unsupported tensor shape: {input.shape}")
        return output

    def __call__(self, features):
        # Determine max length from the batch if max_seq_len is not set
        max_length = features[0]["max_seq_len"]
        if max_length is None:
            max_length = max(len(item["input_ids"]) for item in features)
            logger.warning(f"max_seq_len not set in dataset, using batch maximum length: {max_length}")

        device = features[0]["input_ids"].device

        # Create attention mask that matches the actual input_ids length (not +1)
        batch_attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(len(item["input_ids"]), device=device),
                        torch.zeros(max_length - len(item["input_ids"]), device=device),
                    ]
                ).unsqueeze(0)
                for item in features
            ]
        )
        batch_input_ids = torch.cat([self.padding_tensor(item["input_ids"], max_length) for item in features])
        batch_hidden_states = torch.cat([self.padding_tensor(item["hidden_states"], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [
                torch.cat([item["loss_mask"], torch.zeros(max_length - len(item["loss_mask"]), device=device)]).unsqueeze(0)
                for item in features
            ]
        )

        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask,
        }

        # Only include last_hidden_states if present (not used when use_target_model=True)
        if "last_hidden_states" in features[0]:
            batch_last_hidden_states = torch.cat(
                [self.padding_tensor(item["last_hidden_states"], max_length) for item in features]
            )
            batch["last_hidden_states"] = batch_last_hidden_states

        return batch


class Eagle3TrainerDeepSpeed:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.start_epoch = 0  # Track starting epoch for resume
        self.start_step = 0  # Track starting step for resume
        self.prediction_length = args.prediction_length

        # Load vocabulary mapping
        self._load_vocab_mapping()

        # Load target model if specified
        self.target_model = None
        self.tokenizer = None
        if args.use_target_model:
            self._load_target_model()

        # Parse target model layers
        if args.target_model_layers:
            self.target_model_layers = [int(x.strip()) for x in args.target_model_layers.split(",")]
        else:
            self.target_model_layers = [0, 1, 2]

        # Initialize model and datasets
        self._build_dataloader()
        self._build_model()
        self._initialize_deepspeed()
        self._load_checkpoint()

        # Initialize loss functions
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def _load_target_model(self):
        """Load target model for dynamic hidden state generation"""
        if self.rank == 0:
            logger.info(f"Loading target model from: {self.args.base_model_path}")
            logger.info(f"Target model layers: {self.target_model_layers}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load target model
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path, torch_dtype=dtype, device_map=f"cuda:{self.local_rank}", trust_remote_code=True
        )

        # Set model to eval mode and freeze parameters
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        if self.rank == 0:
            logger.info(f"Target model loaded successfully on device: cuda:{self.local_rank}")
            logger.info(f"Target model vocabulary size: {self.target_model.config.vocab_size}")

    def _load_vocab_mapping(self):
        """Load pre-computed vocabulary mapping"""
        if not os.path.exists(self.args.freq_map_path):
            if self.rank == 0:
                logger.warning(f"Vocabulary mapping file not found: {self.args.freq_map_path}")

        mapping_data = torch.load(self.args.freq_map_path, map_location="cpu")
        self.d2t = mapping_data.get("d2t", None)
        self.t2d = mapping_data.get("t2d", None)

        if self.d2t is None or self.t2d is None:
            raise ValueError(f"Vocabulary mapping not found in {self.args.freq_map_path}, generate it first")

        if self.rank == 0:
            logger.info(f"Loaded vocabulary mapping: draft_vocab_size={len(self.d2t)}, vocab_size={len(self.t2d)}")
            logger.info(f"t2d max value: {torch.max(self.t2d).item()}, min value: {torch.min(self.t2d).item()}")
            logger.info(f"d2t max value: {torch.max(self.d2t).item()}, min value: {torch.min(self.d2t).item()}")

    def _load_checkpoint(self):
        if os.path.exists(os.path.join(self.args.output_dir, "latest")):
            load_path, client_state = self.model_engine.load_checkpoint(
                self.args.output_dir,
                load_optimizer_states=self.args.load_optimizer,
                load_lr_scheduler_states=True,
                load_module_only=False,
            )

            if client_state:
                self.start_epoch = client_state["epoch"]
                self.start_step = client_state["step"]

                if self.rank == 0:
                    logger.info(f"Successfully loaded checkpoint from: {load_path}")
                    logger.info(f"Resuming training from epoch {self.start_epoch} and step {self.start_step}")
                    logger.info(f"Client state keys: {list(client_state.keys())}")
            else:
                if self.rank == 0:
                    logger.info(f"Loaded checkpoint from {load_path}, but no client_state found. Starting from scratch.")

    def _build_model(self):
        # Load model config
        config = AutoConfig.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        config.num_hidden_layers = 1
        config.torch_dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16
        config._attn_implementation = "flash_attention_2"

        # Add Eagle3 specific config
        config.draft_vocab_size = self.args.draft_vocab_size
        config.tie_word_embeddings = False

        # Configure target hidden size based on target model layers
        if self.args.use_target_model and self.target_model:
            target_hidden_size = self.target_model.config.hidden_size * len(self.target_model_layers)
            config.target_hidden_size = target_hidden_size
            if self.rank == 0:
                logger.info(f"Target hidden size: {target_hidden_size} (layers: {self.target_model_layers})")

        # Determine model class - only support Llama for Eagle3
        if hasattr(config, "model_type"):
            if config.model_type.lower() == "llama":
                model_class = LlamaModelEagle3
            elif config.model_type.lower() == "qwen2":
                model_class = Qwen2ModelEagle3
            elif config.model_type.lower() == "qwen3":
                model_class = Qwen3ModelEagle3
            else:
                raise ValueError(f"Eagle3 currently only supports Llama models, got: {config.model_type}")
        else:
            if "llama" in self.args.base_model_path.lower():
                model_class = LlamaModelEagle3
            elif "qwen2" in self.args.base_model_path.lower():
                model_class = Qwen2ModelEagle3
            elif "qwen3" in self.args.base_model_path.lower():
                model_class = Qwen3ModelEagle3
            else:
                raise ValueError("Eagle3 currently only supports Llama and Qwen models")

        self.model = model_class(config=config)

        # Register vocabulary mapping buffers
        self.model.register_buffer("d2t", self.d2t)
        self.model.register_buffer("t2d", self.t2d)

        # Load embeddings and LM head from base model
        self._load_base_model_weights()
        self.model.to(dtype=config.torch_dtype)

    def _load_base_model_weights(self):
        base_path = self.args.base_model_path
        device = f"cuda:{self.local_rank}"
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16

        try:
            # Try loading from safetensors first
            with open(os.path.join(base_path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                try:    # Some models have tie-embed
                    head_path = index_json["weight_map"]["lm_head.weight"]
                except:
                    head_path = None
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]

            with safe_open(os.path.join(base_path, embed_path), framework="pt", device=device) as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                embed_weight = tensor_slice[:, :hidden_dim].to(dtype)

            if head_path is not None:
                with safe_open(os.path.join(base_path, head_path), framework="pt", device=device) as f:
                    tensor_slice = f.get_slice("lm_head.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    lm_head_weight = tensor_slice[:, :hidden_dim].to(dtype)
            else:
                lm_head_weight = embed_weight.clone()

        except:
            # Fallback to pytorch model files
            with open(os.path.join(base_path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                try:    # Some models have tie-embed
                    head_path = index_json["weight_map"]["lm_head.weight"]
                except:
                    head_path = None
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]
            
            embed_weights = torch.load(os.path.join(base_path, embed_path))
            embed_weight = embed_weights["model.embed_tokens.weight"].to(dtype).to(device)

            if head_path is not None:
                head_weights = torch.load(os.path.join(base_path, head_path))
                lm_head_weight = head_weights["lm_head.weight"].to(dtype).to(device)
            else:
                lm_head_weight = embed_weight.clone().to(dtype).to(device)

        # Copy weights to model (embedding tokens only, not LM head as it has different vocab size)
        self.model.embed_tokens.weight.data.copy_(embed_weight)

        # For Eagle3, initialize LM head with subset of original LM head weights
        target_vocab_indices = self.model.d2t
        if len(target_vocab_indices) <= lm_head_weight.shape[0]:
            mapped_lm_head_weight = lm_head_weight[target_vocab_indices]
            self.model.lm_head.weight.data.copy_(mapped_lm_head_weight)
        else:
            # If mapping is larger than available vocab, use available portion
            available_size = min(len(target_vocab_indices), lm_head_weight.shape[0])
            self.model.lm_head.weight.data[:available_size].copy_(lm_head_weight[:available_size])

        # Freeze embedding layer only (LM head needs to be trained for draft vocabulary)
        for param in self.model.embed_tokens.parameters():
            param.requires_grad = False

        if not self.args.use_target_model:
            self.base_model_lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
            self.base_model_lm_head.weight.data.copy_(lm_head_weight)

            for param in self.base_model_lm_head.parameters():
                param.requires_grad = False
        else:
            self.base_model_lm_head = None

        del embed_weight, lm_head_weight
        torch.cuda.empty_cache()
        gc.collect()

    def _build_dataloader(self):
        # First, check if the data path exists
        if not os.path.exists(self.args.data_path):
            raise ValueError(f"Data path does not exist: {self.args.data_path}")

        logger.info(f"Searching for data files in: {self.args.data_path}")

        # Determine max sequence length from directory name if possible
        dataset_max_len = self.args.max_len  # Default max length

        if self.args.use_target_model:
            datapath = []
            for root, dirs, files in os.walk(self.args.data_path):
                for file in files:
                    if file.endswith((".json", ".pt", ".ckpt")):
                        file_path = os.path.join(root, file)
                        datapath.append(file_path)

            if not datapath:
                raise ValueError(f"No suitable data files found in {self.args.data_path}")

            if self.rank == 0:
                logger.info(f"Found {len(datapath)} data files for target model mode")
                logger.info("Example file paths:")
                for path in datapath[:3]:
                    logger.info(f"  {path}")
        else:
            # Original behavior: look for pre-computed hidden state files
            # List all subdirectories
            dir_list = [d for d in os.listdir(self.args.data_path) if os.path.isdir(os.path.join(self.args.data_path, d))]
            dir_list.sort()
            logger.info(f"Found directories: {dir_list}")

            for dirname in dir_list:
                if "-" in dirname:
                    try:
                        max_k_value = dirname.split("-")[1].strip()
                        if max_k_value.endswith("K"):
                            dataset_max_len = int(float(max_k_value[:-1]) * 1024)
                        else:
                            dataset_max_len = int(max_k_value)
                        break
                    except:
                        continue

            logger.info(f"Using max sequence length: {dataset_max_len}")

            datapath = []
            for root, dirs, files in os.walk(self.args.data_path):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".ckpt"):
                        file_path = os.path.join(root, file)
                        datapath.append(file_path)

            if not datapath:
                raise ValueError(f"No .ckpt/.pt files found in {self.args.data_path} or its subdirectories")

            logger.info(f"Total number of .ckpt/.pt files found: {len(datapath)}")

        # Split into train and validation
        train_size = int(len(datapath) * 0.95)
        train_files = datapath[:train_size]
        val_files = datapath[train_size:]
        print(f"train_size: {train_size}, val_size: {len(val_files)}")

        # Create datasets with proper parameters
        self.train_dataset = EagleDataset(
            train_files,
            dataset_max_len=dataset_max_len,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            use_target_model=self.args.use_target_model,
            target_model_layers=self.target_model_layers,
        )
        self.val_dataset = EagleDataset(
            val_files,
            dataset_max_len=dataset_max_len,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            use_target_model=self.args.use_target_model,
            target_model_layers=self.target_model_layers,
        )

        if self.rank == 0:
            logger.info(f"Found {len(train_files)} training files and {len(val_files)} validation files")
            # Log some example file paths for verification
            logger.info("Example training file paths:")
            for path in train_files[:3]:
                logger.info(f"  {path}")
            if val_files:
                logger.info("Example validation file paths:")
                for path in val_files[:3]:
                    logger.info(f"  {path}")

        # Verify data loading by trying to load the first file
        if train_files:
            try:
                first_item = self.train_dataset[0]
                logger.info(f"Successfully loaded first data item")
                logger.info(f"Data keys: {list(first_item.keys())}")
                if "input_ids" in first_item:
                    logger.info(f"First item input_ids length: {len(first_item['input_ids'])}")
                if "hidden_states" in first_item:
                    logger.info(f"First item hidden_states shape: {first_item['hidden_states'].shape}")
            except Exception as e:
                logger.error(f"Failed to load first data item: {str(e)}")
                raise

    def _initialize_deepspeed(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        with open(self.args.deepspeed_config) as f:
            ds_config = json.load(f)

        ds_config["train_micro_batch_size_per_gpu"] = self.args.batch_size
        ga_steps = ds_config.get("gradient_accumulation_steps", 1)
        ds_config["train_batch_size"] = self.args.batch_size * ga_steps * dist.get_world_size()
        logger.info(
            f"Overide train_micro_batch_size_per_gpu: {ds_config['train_micro_batch_size_per_gpu']}, train_batch_size: {ds_config['train_batch_size']}"
        )

        # Initialize DeepSpeed engine - don't pass args to avoid config conflict
        self.model_engine, self.optimizer, self.train_loader, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=parameters,
            training_data=self.train_dataset,
            collate_fn=EagleDataCollator(),
            config=ds_config,
        )

        if not self.args.use_target_model and self.base_model_lm_head is not None:
            self.base_model_engine = self.base_model_lm_head.to(dtype=self.model.config.torch_dtype).to(
                device=f"cuda:{self.local_rank}"
            )
            self.base_model_engine.requires_grad_(False)
        else:
            self.base_model_engine = None

        val_sampler = DistributedSampler(
            self.val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
        )

        # Create validation dataloader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size * 2,
            shuffle=False,
            collate_fn=EagleDataCollator(),
            num_workers=4,
            sampler=val_sampler,
            pin_memory=True,
        )

    @torch.no_grad()
    def _padding(self, tensor, left=True):
        """Utility function to pad tensors as used in Eagle3"""
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        else:
            tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
        return tensor

    def _compute_loss(self, batch):
        """Compute Eagle3 multi-step prediction losses"""
        input_ids = batch["input_ids"]
        hidden_states = batch["hidden_states"]  # Pre-cached hidden states from target model or generated dynamically
        last_hidden_states = batch.get("last_hidden_states", None)  # May not exist when use_target_model=True
        loss_mask = batch["loss_mask"]
        attention_mask = batch["attention_mask"]
        batch_size, seq_length = input_ids.shape

        # Prepare target logits
        with torch.no_grad():
            if self.args.use_target_model and self.target_model:
                # Clear any existing cache in target model to prevent accumulation
                if hasattr(self.target_model, "past_key_values"):
                    self.target_model.past_key_values = None

                if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                    start_token_id = self.tokenizer.bos_token_id
                elif hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    start_token_id = self.tokenizer.eos_token_id
                else:
                    start_token_id = 1  # Default fallback

                full_input_ids = torch.cat(
                    [torch.full((batch_size, 1), start_token_id, dtype=input_ids.dtype, device=input_ids.device), input_ids],
                    dim=1,
                )
                full_attention_mask = torch.cat(
                    [
                        torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[:, :-1],  # Remove last padding
                    ],
                    dim=1,
                )

                # Generate target model outputs - explicitly disable cache to prevent accumulation
                target_outputs = self.target_model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,  # Explicitly disable cache to prevent memory accumulation
                    past_key_values=None,  # Ensure no cache is used
                )
                target_logits = target_outputs.logits
                target_logits = target_logits[:, 1:, :]  # Remove first token, align with current sequence
            else:
                if last_hidden_states is None:
                    raise ValueError("last_hidden_states is required when use_target_model=False")
                target_logits = self.base_model_engine(last_hidden_states.to(dtype=self.model.config.torch_dtype))
                target_logits = self._padding(target_logits, left=False)

            loss_mask = loss_mask[..., None]

        # Move target_logits to same device/dtype before model forward
        target_logits = target_logits.to(device=hidden_states.device, dtype=self.model.config.torch_dtype)

        loss_list, accuracy_list = self.model_engine(
            base_model_hidden_states=hidden_states.to(dtype=self.model.config.torch_dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=True,
            prediction_length=self.prediction_length,
            target=target_logits,
            loss_mask=loss_mask,
        )

        # Clean up large tensors to free memory
        del target_logits

        return loss_list, accuracy_list

    def _validate_epoch(self, epoch, tracking):
        """Run validation after each training epoch"""
        if self.rank == 0:
            logger.info(f"Starting validation for epoch {epoch + 1}")

        # Set model to evaluation mode
        self.model_engine.eval()

        # Track validation metrics for each prediction step
        val_loss_list = [[] for _ in range(self.prediction_length)]
        val_accuracy_list = [[] for _ in range(self.prediction_length)]

        if self.rank == 0:
            val_iter = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
        else:
            val_iter = self.val_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter):
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["hidden_states"] = batch["hidden_states"].to(dtype=self.model.config.torch_dtype)

                loss_list, accuracy_list = self._compute_loss(batch)

                # Store metrics for each prediction step
                for i in range(len(loss_list)):
                    val_loss_list[i].append(loss_list[i].item())
                    val_accuracy_list[i].append(accuracy_list[i])

                if self.rank == 0:
                    # Show average accuracy in progress bar
                    avg_acc = sum(accuracy_list) / len(accuracy_list)
                    val_iter.set_postfix(
                        {"val_loss": f"{sum([l.item() for l in loss_list]):.4f}", "val_avg_acc": f"{avg_acc:.2%}"}
                    )

        # Aggregate validation metrics across all processes
        for i in range(self.prediction_length):
            # Convert to tensors for distributed reduction
            avg_val_loss = torch.tensor(val_loss_list[i]).cuda().mean()
            avg_val_acc = torch.tensor(val_accuracy_list[i]).cuda().mean()

            # All-reduce across processes
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_val_acc, op=dist.ReduceOp.AVG)

            avg_val_loss = avg_val_loss.item()
            avg_val_acc = avg_val_acc.item()

            # Log validation metrics
            if self.rank == 0 and tracking:
                global_step = epoch * len(self.train_loader) + len(self.train_loader) - 1
                tracking.log(
                    {
                        f"val/ploss_{i}": avg_val_loss,
                        f"val/acc_{i}": avg_val_acc,
                        f"val/epoch_ploss_{i}": avg_val_loss,
                        f"val/epoch_acc_{i}": avg_val_acc,
                    },
                    step=global_step,
                )

                logger.info(
                    f"Validation Epoch [{epoch + 1}/{self.args.epochs}], Step {i}, pLoss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2%}"
                )

        # Set model back to training mode
        self.model_engine.train()

    def train(self):
        if self.rank == 0:
            tracking = Tracking(
                project_name=self.args.project_name, experiment_name=self.args.experiment_name, default_backend="wandb"
            )
        else:
            tracking = None

        log_memory_stats(self.rank, "Training Start")

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model_engine.train()

            if self.rank == 0:
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                train_iter = self.train_loader

            # Track epoch metrics for each prediction step
            epoch_loss_list = [[] for _ in range(self.prediction_length)]
            epoch_accuracy_list = [[] for _ in range(self.prediction_length)]

            steps_per_epoch = len(self.train_loader)
            for batch_idx, batch in enumerate(train_iter):
                global_step = epoch * steps_per_epoch + batch_idx
                if global_step < self.start_step:
                    continue

                self.model_engine.zero_grad()
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["hidden_states"] = batch["hidden_states"].to(dtype=self.model.config.torch_dtype)

                loss_list, accuracy_list = self._compute_loss(batch)

                loss_weights = [0.8**i for i in range(len(loss_list))]
                total_loss = sum([loss_weights[i] * loss_list[i] for i in range(len(loss_list))])

                self.model_engine.backward(total_loss)
                self.model_engine.step()

                # Store metrics for each prediction step
                for i in range(len(loss_list)):
                    epoch_loss_list[i].append(loss_list[i].item())
                    epoch_accuracy_list[i].append(accuracy_list[i])

                if self.rank == 0:
                    metrics = {
                        "train/total_loss": total_loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }

                    # Log metrics for each prediction step
                    for i in range(len(loss_list)):
                        metrics[f"train/ploss_{i}"] = loss_list[i].item()
                        metrics[f"train/acc_{i}"] = accuracy_list[i]

                    tracking.log(metrics, step=global_step)

                    # Show average accuracy in progress bar
                    avg_acc = sum(accuracy_list) / len(accuracy_list)
                    train_iter.set_postfix({"loss": f"{total_loss.item():.4f}", "avg_acc": f"{avg_acc:.2%}", "epoch": epoch})

                # Checkpointing logic
                if self.args.save_steps and (global_step + 1) % self.args.save_steps == 0:
                    client_state = {
                        "epoch": epoch,
                        "step": global_step + 1,
                    }
                    log_memory_stats(self.rank, f"Before Cleanup Step {global_step + 1}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    log_memory_stats(self.rank, f"After Cleanup Step {global_step + 1}")
                    self.model_engine.save_checkpoint(self.args.output_dir, client_state=client_state, exclude_frozen_parameters=False)
                    if self.rank == 0:
                        logger.info(f"Checkpoint saved at step {global_step + 1}")

            # Log epoch averages
            if self.rank == 0:
                for i in range(self.prediction_length):
                    avg_ploss = sum(epoch_loss_list[i]) / len(epoch_loss_list[i])
                    avg_acc = sum(epoch_accuracy_list[i]) / len(epoch_accuracy_list[i])

                    logger.info(
                        f"Train Epoch [{epoch + 1}/{self.args.epochs}], Step {i}, pLoss: {avg_ploss:.4f}, Acc: {avg_acc:.2%}"
                    )

                    tracking.log(
                        {
                            f"train/epoch_ploss_{i}": avg_ploss,
                            f"train/epoch_acc_{i}": avg_acc,
                        },
                        step=epoch * len(self.train_loader) + len(self.train_loader) - 1,
                    )

            self._validate_epoch(epoch, tracking)

            if not self.args.save_steps:
                client_state = {
                    "epoch": epoch + 1,
                    "step": (epoch + 1) * len(self.train_loader),
                }

                # Clear cache before checkpoint saving to free memory
                log_memory_stats(self.rank, f"Before Cleanup Epoch {epoch}")
                torch.cuda.empty_cache()
                gc.collect()
                log_memory_stats(self.rank, f"After Cleanup Epoch {epoch}")

                self.model_engine.save_checkpoint(self.args.output_dir, client_state=client_state, exclude_frozen_parameters=False)

                if self.rank == 0:
                    logger.info(f"Checkpoint saved at epoch {epoch}: {self.args.output_dir}")


def main():
    parser = add_args()
    args = parser.parse_args()

    # Set environment variables to reduce DeepSpeed verbosity
    os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "WARNING")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

    # Initialize distributed training
    deepspeed.init_distributed()

    # Create trainer and start training
    trainer = Eagle3TrainerDeepSpeed(args)
    trainer.train()


if __name__ == "__main__":
    main()
