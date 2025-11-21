import numbers
import os
from datetime import timedelta
from typing import Dict


def initialize_global_process_group(timeout_second=36000, spmd=False):
    import torch.distributed

    if not torch.distributed.is_initialized():  # Check if already initialized
        print("Initializing process group...")
        torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))
    else:
        print("Process group already initialized.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not CUDA_VISIBLE_DEVICES:
        if spmd:
            # CUDA_VISIBLE_DEVICES = ','.join(str(i) for i in range(tensor_parallel_size))
            CUDA_VISIBLE_DEVICES = ",".join(str(i) for i in range(world_size))
        else:
            CUDA_VISIBLE_DEVICES = str(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"CUDA_VISIBLE_DEVICES is not set, set to {CUDA_VISIBLE_DEVICES}")

    return local_rank, rank, world_size


def concat_dict_to_str(dict: Dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{v:.3f}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:

    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        if print_to_console:
            print("Using LocalLogger is deprecated. The constructor API will change ")

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class Tracking:
    supported_backend = ["wandb", "console"]

    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]

        for backend in default_backend:
            assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "console" in default_backend:
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
