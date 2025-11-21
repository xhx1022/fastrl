import asyncio
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

import torch.distributed as dist
import zmq
import zmq.asyncio
import zmq.error
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def get_free_port() -> int:
    """Get a free port for ZMQ communication."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_host_ip():
    """Get host IP from environment variables or network interfaces."""
    host_ipv4 = os.environ.get("MY_HOST_IP", None)
    host_ipv6 = os.environ.get("MY_HOST_IPV6", None)
    if host_ipv4 or host_ipv6:
        return host_ipv4 or host_ipv6

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
        if host_ip and host_ip != "127.0.0.1":
            logger.info(f"Detected network-accessible IP: {host_ip}")
            return host_ip
    except Exception as e:
        logger.debug(f"Could not auto-detect network IP: {e}")

    logger.warning(
        "Could not determine network-accessible IP. Using 127.0.0.1. "
        "For multi-node setups, set MY_HOST_IP environment variable."
    )
    return "127.0.0.1"


class WorkerState(Enum):
    """Worker state machine."""

    IDLE = "idle"
    GENERATING = "generating"
    RELEASED = "released"
    TRAINING = "training"
    COMPLETED = "completed"


@dataclass
class WorkerInfo:
    worker_id: int
    gpu_id: int
    dp_rank: int
    tp_rank: int
    state: WorkerState
    last_update: float = 0.0


class CoordinatorEvent(Enum):
    """Events broadcast by coordinator to workers."""

    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    BATCH_COMPLETE = "batch_complete"


@dataclass
class TrainingCommand:
    """Command to start training with specific worker configuration."""

    event: CoordinatorEvent
    training_ranks: List[int]  # Process ranks that should train
    timestamp: float


class CentralCoordinator:
    """Central coordinator using PUB-SUB pattern for event broadcasting."""

    def __init__(self, req_port: int, pub_port: int, tp_size: int = 1, min_workers_for_training: int = 1):
        self.context = zmq.asyncio.Context()

        # REP socket for worker requests (register, release, mark_completed)
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.rep_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.rep_socket.setsockopt(zmq.LINGER, 0)
        self.rep_socket.bind(f"tcp://*:{req_port}")

        # PUB socket for broadcasting events to workers
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.bind(f"tcp://*:{pub_port}")

        # State
        self.worker_states: Dict[int, WorkerInfo] = {}
        self.tp_size = tp_size
        self.min_workers_for_training = min_workers_for_training
        self._stopped = False

        # Training state
        self.training_active = False
        self.training_ranks: Set[int] = set()

        logger.info(
            f"Coordinator started: REQ port={req_port}, PUB port={pub_port}, "
            f"tp_size={tp_size}, min_workers={min_workers_for_training}"
        )

    async def run(self):
        """Main coordinator loop."""
        logger.info("Coordinator run loop starting")

        # Give pub socket time to establish connections
        await asyncio.sleep(0.5)

        while not self._stopped:
            try:
                poller = zmq.asyncio.Poller()
                poller.register(self.rep_socket, zmq.POLLIN)
                socks = dict(await poller.poll(10))

                if self.rep_socket in socks:
                    message = await self.rep_socket.recv_json()
                    response = await self._process_request(message)
                    await self.rep_socket.send_json(response)
                else:
                    await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Coordinator error: {e}", exc_info=True)
                try:
                    await self.rep_socket.send_json({"status": "error", "message": str(e)})
                except Exception:
                    pass

        logger.info("Coordinator run loop exiting")
        self.rep_socket.close(0)
        self.pub_socket.close(0)
        self.context.term()

    def stop(self):
        """Signal the run loop to stop."""
        self._stopped = True

    async def _process_request(self, message: dict) -> dict:
        """Process incoming request and return response."""
        request_type = message.get("type")
        worker_id = message.get("worker_id")

        if request_type == "register":
            worker_info = WorkerInfo(
                worker_id=worker_id,
                gpu_id=message["gpu_id"],
                dp_rank=message["dp_rank"],
                tp_rank=message["tp_rank"],
                state=WorkerState.IDLE,
                last_update=time.time(),
            )
            self.worker_states[worker_id] = worker_info
            logger.info(f"Registered worker {worker_id} (dp={message['dp_rank']}, tp={message['tp_rank']})")
            return {"status": "ok"}

        elif request_type == "start_generation":
            if worker_id in self.worker_states:
                self.worker_states[worker_id].state = WorkerState.GENERATING
                self.worker_states[worker_id].last_update = time.time()
            return {"status": "ok"}

        elif request_type == "release":
            if worker_id not in self.worker_states:
                return {"status": "error", "message": "Worker not found"}

            worker_info = self.worker_states[worker_id]
            dp_rank = worker_info.dp_rank

            # Mark all TP ranks for this DP worker as released
            for wid, winfo in self.worker_states.items():
                if winfo.dp_rank == dp_rank:
                    self.worker_states[wid].state = WorkerState.RELEASED
                    self.worker_states[wid].last_update = time.time()

            logger.info(f"Worker {worker_id} (DP={dp_rank}) released all TP ranks")

            # Check if we should start training
            await self._check_and_start_training()

            return {"status": "ok"}

        elif request_type == "mark_completed":
            if worker_id in self.worker_states:
                self.worker_states[worker_id].state = WorkerState.COMPLETED
                self.worker_states[worker_id].last_update = time.time()

                # Check if this worker was training
                worker_was_training = worker_id in self.training_ranks

                # Check if all workers completed
                all_completed = all(w.state == WorkerState.COMPLETED for w in self.worker_states.values())

                # Stop training if: (1) any training worker completes OR (2) all workers complete
                should_stop_training = self.training_active and (worker_was_training or all_completed)

                if should_stop_training:
                    if worker_was_training:
                        logger.info(
                            f"Training worker {worker_id} completed, broadcasting STOP_TRAINING "
                            f"to prevent blocking next batch"
                        )
                    else:
                        logger.info("All workers completed, broadcasting STOP_TRAINING and BATCH_COMPLETE")
                    await self._broadcast_stop_training()

                if all_completed:
                    logger.info("All workers completed, broadcasting BATCH_COMPLETE")
                    await self._broadcast_event(CoordinatorEvent.BATCH_COMPLETE, [])

                return {"status": "ok", "all_completed": all_completed}

            return {"status": "error", "message": "Worker not found"}

        elif request_type == "get_state":
            return {
                "status": "ok",
                "training_active": self.training_active,
                "training_ranks": list(self.training_ranks),
            }

        return {"status": "error", "message": "Unknown request type"}

    async def _check_and_start_training(self):
        """Check if conditions are met to start training and broadcast command if so."""
        if self.training_active:
            return  # Training already active

        # Count released DP workers
        released_dp_ranks = set()
        for winfo in self.worker_states.values():
            if winfo.state == WorkerState.RELEASED:
                released_dp_ranks.add(winfo.dp_rank)

        if len(released_dp_ranks) < self.min_workers_for_training:
            return  # Not enough workers released

        # Select which DP workers will train (first N released)
        selected_dp_ranks = sorted(released_dp_ranks)[: self.min_workers_for_training]

        # Get all process ranks (including all TP ranks) for selected DP workers
        training_ranks = []
        for wid, winfo in self.worker_states.items():
            if winfo.dp_rank in selected_dp_ranks:
                training_ranks.append(wid)

        training_ranks = sorted(training_ranks)

        # Mark workers as training
        for wid in training_ranks:
            self.worker_states[wid].state = WorkerState.TRAINING
            self.training_ranks.add(wid)

        self.training_active = True

        logger.info(
            f"Starting training: DP ranks {selected_dp_ranks}, "
            f"process ranks {training_ranks} ({len(training_ranks)} total)"
        )

        # Broadcast START_TRAINING event to all workers
        await self._broadcast_event(CoordinatorEvent.START_TRAINING, training_ranks)

    async def _broadcast_stop_training(self):
        """Broadcast stop training command."""
        if not self.training_active:
            return

        logger.info("Broadcasting STOP_TRAINING event")
        await self._broadcast_event(CoordinatorEvent.STOP_TRAINING, [])

        self.training_active = False
        self.training_ranks.clear()

    async def _broadcast_event(self, event: CoordinatorEvent, training_ranks: List[int]):
        """Broadcast an event to all subscribed workers."""
        command = {
            "event": event.value,
            "training_ranks": training_ranks,
            "timestamp": time.time(),
        }

        # Send multiple times to ensure delivery (PUB-SUB is fire-and-forget)
        for _ in range(3):
            await self.pub_socket.send_json(command)
            await asyncio.sleep(0.05)

    async def cleanup(self):
        """Signal coordinator loop to stop."""
        self.stop()


class WorkerClient:
    """Client for workers to communicate with coordinator."""

    def __init__(self, req_address: str, sub_address: str, worker_id: int):
        self.worker_id = worker_id
        self.context = zmq.asyncio.Context()

        # REQ socket for sending requests to coordinator
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(req_address)
        self.req_socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.req_socket.setsockopt(zmq.SNDTIMEO, 10000)

        # SUB socket for receiving broadcasts from coordinator
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(sub_address)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self._lock = asyncio.Lock()

        logger.info(f"Worker {worker_id} client initialized: REQ={req_address}, SUB={sub_address}")

    async def register_worker(self, worker_info: dict) -> dict:
        """Register worker with coordinator."""
        async with self._lock:
            request = {"type": "register", **worker_info}
            await self.req_socket.send_json(request)
            return await self.req_socket.recv_json()

    async def start_generation(self, worker_id: int) -> dict:
        """Notify coordinator that generation started."""
        async with self._lock:
            request = {"type": "start_generation", "worker_id": worker_id}
            await self.req_socket.send_json(request)
            return await self.req_socket.recv_json()

    async def release_worker(self, worker_id: int) -> dict:
        """Notify coordinator that worker released memory."""
        async with self._lock:
            request = {"type": "release", "worker_id": worker_id}
            try:
                await self.req_socket.send_json(request)
                return await self.req_socket.recv_json()
            except zmq.error.ZMQError as e:
                logger.warning(f"Worker {worker_id} ZMQ error releasing: {e}, continuing anyway")
                return {"status": "ok"}

    async def mark_completed(self, worker_id: int) -> dict:
        """Mark worker as completed generation."""
        async with self._lock:
            request = {"type": "mark_completed", "worker_id": worker_id}
            try:
                await self.req_socket.send_json(request)
                return await self.req_socket.recv_json()
            except zmq.error.ZMQError as e:
                logger.warning(f"Worker {worker_id} ZMQ error marking completed: {e}, treating as completed")
                return {"status": "ok", "all_completed": False}

    async def get_state(self) -> dict:
        """Get current coordinator state."""
        async with self._lock:
            request = {"type": "get_state"}
            await self.req_socket.send_json(request)
            return await self.req_socket.recv_json()

    async def wait_for_event(self, timeout: float = 1.0) -> Optional[TrainingCommand]:
        """Wait for broadcast event from coordinator (non-blocking with timeout)."""
        try:
            message = await asyncio.wait_for(self.sub_socket.recv_json(), timeout=timeout)
            event = CoordinatorEvent(message["event"])
            return TrainingCommand(
                event=event,
                training_ranks=message["training_ranks"],
                timestamp=message["timestamp"],
            )
        except asyncio.TimeoutError:
            return None
        except zmq.error.Again:
            return None
        except Exception as e:
            logger.debug(f"Worker {self.worker_id} error receiving event: {e}")
            return None

    async def cleanup(self):
        """Clean up resources."""
        try:
            self.req_socket.close()
            self.sub_socket.close()
        except Exception as e:
            logger.debug(f"Error closing sockets: {e}")
        try:
            self.context.term()
        except Exception as e:
            logger.debug(f"Error terminating context: {e}")


class RolloutDrafterManager:
    """Manages early memory release and coordinates background training on freed GPUs."""

    def __init__(self, device_mesh: DeviceMesh, rollout_config):
        self.device_mesh = device_mesh
        self.rollout_config = rollout_config

        assert dist.is_initialized()

        # Ranks and mesh info
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tp_size = device_mesh["tp"].size()
        self.dp_size = device_mesh["dp"].size()
        self.dp_rank = device_mesh["dp"].get_local_rank()
        self.tp_rank = device_mesh["tp"].get_local_rank()

        # Training configuration
        if (
            hasattr(rollout_config, "speculative")
            and hasattr(rollout_config.speculative, "train")
            and hasattr(rollout_config.speculative.train, "min_workers_for_training")
        ):
            self.min_workers_for_training = rollout_config.speculative.train.min_workers_for_training
        else:
            self.min_workers_for_training = 1

        # Coordinator setup
        self.coordinator = None
        self.worker_client = None
        self.is_coordinator = self.rank == 0
        self._coord_thread: Optional[threading.Thread] = None

        # Background training
        self.train_drafter = (
            hasattr(rollout_config, "speculative")
            and rollout_config.speculative.get("enable", False)
            and rollout_config.speculative.get("train", {}).get("enable_drafter_training", False)
        )
        self.background_trainer = None

        # RL step tracking
        self.current_rl_step = 0
        self.training_interval_steps = (
            rollout_config.speculative.get("train", {}).get("training_interval_steps", 1)
            if hasattr(rollout_config, "speculative") and rollout_config.speculative.get("enable", False)
            else 1
        )

        # State
        self._training_task: Optional[asyncio.Task] = None
        self._event_listener_task: Optional[asyncio.Task] = None
        self._training_active = False
        self._training_stop_event = threading.Event()  # Thread-safe stop signal
        self._training_cleanup_complete = threading.Event()  # Signals when training cleanup is done
        self._training_cleanup_complete.set()  # Initially set (no training to cleanup)

        # Local worker info
        self.gpu_id = self.rank
        self.hostname = os.environ.get("HOSTNAME", "unknown")

        # Device meshes for drafter
        self.global_device_mesh_list = [
            DeviceMesh("cuda", list(range(i * self.tp_size, (i + 1) * self.tp_size))) for i in range(self.dp_size)
        ]
        self.drafter_device_mesh = self.global_device_mesh_list[self.dp_rank]

        logger.info(
            f"RolloutDrafterManager initialized: rank={self.rank}, "
            f"tp_size={self.tp_size}, dp_size={self.dp_size}, "
            f"min_workers_for_training={self.min_workers_for_training}"
        )

    def should_train_this_step(self) -> bool:
        if not self.train_drafter:
            return False
        return self.current_rl_step % self.training_interval_steps == 0

    def should_collect_data_this_step(self) -> bool:
        """Check if we should collect data this step (one step before training)."""
        if not self.train_drafter:
            return False
        return (self.current_rl_step + 1) % self.training_interval_steps == 0

    def increment_rl_step(self):
        """Increment the RL step counter."""
        self.current_rl_step += 1
        logger.debug(f"RolloutDrafterManager RL step incremented to {self.current_rl_step}")

    async def initialize(self):
        """Initialize communication system."""
        if self.is_coordinator:
            # Start coordinator in dedicated thread
            req_port = get_free_port()
            pub_port = get_free_port()
            self.coordinator = CentralCoordinator(
                req_port=req_port,
                pub_port=pub_port,
                tp_size=self.tp_size,
                min_workers_for_training=self.min_workers_for_training,
            )

            def _run_coordinator_loop():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.coordinator.run())
                except Exception as e:
                    logger.exception(f"Coordinator thread crashed: {e}")

            self._coord_thread = threading.Thread(target=_run_coordinator_loop, daemon=True)
            self._coord_thread.start()

            await asyncio.sleep(0.5)

            # Broadcast coordinator addresses
            req_address = f"tcp://{get_host_ip()}:{req_port}"
            pub_address = f"tcp://{get_host_ip()}:{pub_port}"
            logger.info(f"Coordinator started: REQ={req_address}, PUB={pub_address}")
        else:
            req_address = None
            pub_address = None

        # Broadcast addresses to all workers
        addresses = [req_address, pub_address]
        dist.broadcast_object_list(addresses, src=0)
        req_address, pub_address = addresses

        # Create worker client
        self.worker_client = WorkerClient(req_address, pub_address, self.rank)

        if not self.is_coordinator:
            await asyncio.sleep(0.5)

        # Register this worker
        await self.worker_client.register_worker(
            {
                "worker_id": self.rank,
                "gpu_id": self.gpu_id,
                "dp_rank": self.dp_rank,
                "tp_rank": self.tp_rank,
            }
        )

        logger.info(f"Worker {self.rank} registered with coordinator")

        # Start event listener in separate thread (not asyncio task)
        self._event_listener_running = True
        self._event_listener_thread = threading.Thread(target=self._event_listener_thread_func, daemon=True)
        self._event_listener_thread.start()

    def _event_listener_thread_func(self):
        """Thread function to listen for events and coordinate state (runs in separate thread)."""
        logger.info(f"Worker {self.rank} event listener thread started")

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        check_counter = 0

        while self._event_listener_running:
            try:
                # Check for broadcasts (non-blocking with timeout)
                command = loop.run_until_complete(self.worker_client.wait_for_event(timeout=0.5))

                if command is None:
                    # No broadcast, do periodic state check every 2.5 seconds
                    check_counter += 1
                    if check_counter >= 5:
                        check_counter = 0
                        if not self._training_active:
                            loop.run_until_complete(self._check_training_state_sync(loop))
                    time.sleep(0.001)  # Reduced from 0.01 to 0.001 for faster response
                    continue

                check_counter = 0
                logger.debug(f"Worker {self.rank} received event: {command.event.value}")

                if command.event == CoordinatorEvent.START_TRAINING:
                    if self.rank in command.training_ranks:
                        logger.debug(f"Worker {self.rank} is in training group, starting training")
                        loop.run_until_complete(self._start_training(command.training_ranks))
                    else:
                        logger.debug(f"Worker {self.rank} not in training group, continuing")

                elif command.event == CoordinatorEvent.STOP_TRAINING:
                    logger.debug(f"Worker {self.rank} received STOP_TRAINING, stopping if training")
                    loop.run_until_complete(self._stop_training())

                elif command.event == CoordinatorEvent.BATCH_COMPLETE:
                    logger.debug(f"Worker {self.rank} batch complete")

            except Exception as e:
                logger.exception(f"Worker {self.rank} event listener thread error: {e}")
                time.sleep(0.1)

        logger.info(f"Worker {self.rank} event listener thread exiting")
        loop.close()

    async def _check_training_state_sync(self, loop):
        """Check coordinator state and start training if needed (runs in event listener thread)."""
        try:
            response = await self.worker_client.get_state()
            if response["status"] != "ok":
                return

            training_active = response.get("training_active", False)
            training_ranks = response.get("training_ranks", [])

            if training_active and self.rank in training_ranks and not self._training_active:
                logger.info(
                    f"Worker {self.rank} detected it should be training (state check fallback), "
                    f"starting training with ranks {training_ranks}"
                )
                await self._start_training(training_ranks)

        except Exception as e:
            logger.debug(f"Worker {self.rank} error checking training state: {e}")

    async def start_generation(self):
        """Notify coordinator that generation started."""
        await self.worker_client.start_generation(self.rank)

    async def release_worker_memory(self, worker_id: int) -> bool:
        """Release memory and notify coordinator."""
        if worker_id != self.rank:
            logger.warning(f"Worker {self.rank} cannot release memory for worker {worker_id}")
            return False

        # Check if should train this RL step
        if not self.should_train_this_step():
            logger.debug(
                f"Worker {self.rank} skipping training for RL step {self.current_rl_step} "
                f"(interval={self.training_interval_steps})"
            )
            return True

        # Notify coordinator
        response = await self.worker_client.release_worker(worker_id)

        if response["status"] != "ok":
            logger.error(f"Failed to release worker {worker_id}: {response}")
            return False

        logger.debug(f"Worker {worker_id} memory released successfully")
        return True

    async def mark_worker_completed(self, worker_id: int):
        """Mark worker as completed generation."""
        response = await self.worker_client.mark_completed(worker_id)

        if response["status"] == "ok":
            all_completed = response.get("all_completed", False)
            logger.debug(f"Worker {worker_id} marked as completed. All completed: {all_completed}")

    async def _start_training(self, training_ranks: List[int]):
        """Start training on this worker (non-blocking initialization)."""
        if self._training_active:
            logger.debug(f"Worker {self.rank} already training, ignoring start command")
            return

        if self.background_trainer is None:
            logger.warning(f"Worker {self.rank} has no background_trainer")
            return

        logger.debug(f"Worker {self.rank} starting training (non-blocking)")

        # Clear stop event for new training session
        self._training_stop_event.clear()

        # Clear cleanup complete event - training is starting
        self._training_cleanup_complete.clear()

        # Set active immediately to prevent double initialization
        self._training_active = True

        # Start training loop that handles initialization internally
        self._training_task = asyncio.create_task(self._run_training_loop_with_init(training_ranks))

    async def _run_training_loop_with_init(self, training_ranks: List[int]):
        """Training loop with initialization in background."""
        logger.debug(f"Worker {self.rank} initializing training model in background")

        try:
            # Initialize training model (this is the 30s blocking operation)
            success = await asyncio.wait_for(
                self.background_trainer.activate_training_model(self.drafter_device_mesh, training_ranks),
                timeout=30.0,
            )

            if not success:
                logger.error(f"Worker {self.rank} failed to activate training model")
                self._training_active = False
                return

            self.background_trainer.set_training_active(True)
            logger.debug(f"Worker {self.rank} training model activated, starting training loop")

            # Run actual training loop
            await self._run_training_loop()

        except Exception as e:
            logger.exception(f"Worker {self.rank} error in training initialization: {e}")
            self._training_active = False

    async def _run_training_loop(self):
        """Simple linear training loop."""
        logger.debug(f"Worker {self.rank} training loop started")

        try:
            step = 0
            max_steps = 200

            while self._training_active and step < max_steps:
                # Check thread-safe stop event (set by event listener thread)
                if self._training_stop_event.is_set():
                    logger.info(f"Worker {self.rank} received stop signal at step {step}")
                    break

                # Check if training should stop (simple flag check, no polling)
                if not self.background_trainer.is_training_active:
                    logger.info(f"Worker {self.rank} training stopped by background_trainer flag at step {step}")
                    break

                # Execute training step
                success = await asyncio.wait_for(
                    self.background_trainer.training_step(step),
                    timeout=20.0,
                )

                step += 1
                if success:
                    step += 1
                else:
                    logger.debug(f"Worker {self.rank} training is not ready at step {step}")

                await asyncio.sleep(0.02)

            logger.debug(f"Worker {self.rank} training loop completed: {step} steps")

        except Exception as e:
            logger.exception(f"Worker {self.rank} training loop error: {e}")

        finally:
            # Cleanup
            logger.info(f"Worker {self.rank} cleaning up training")
            await self.background_trainer.cleanup_training()
            self._training_active = False
            # Signal that cleanup is complete
            self._training_cleanup_complete.set()
            logger.debug(f"Worker {self.rank} training cleanup complete")

    async def _stop_training(self):
        """Stop training on this worker."""
        if not self._training_active:
            return

        logger.info(f"Worker {self.rank} stopping training")

        # Set thread-safe stop event (training loop checks this)
        self._training_stop_event.set()

        # Signal training to stop
        self._training_active = False
        if self.background_trainer:
            self.background_trainer.set_training_active(False)

        # Wait for training task to complete
        if self._training_task and not self._training_task.done():
            try:
                await asyncio.wait_for(self._training_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Worker {self.rank} training task did not stop within timeout")
                self._training_task.cancel()

        # Ensure cleanup complete event is set
        logger.debug(f"Worker {self.rank} _stop_training completed, signaling cleanup done")
        self._training_cleanup_complete.set()

    def get_global_release_status(self) -> dict:
        """Get current status for logging."""
        return {
            "current_worker": self.rank,
            "training_active": self._training_active,
        }
