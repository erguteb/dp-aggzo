"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import (
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_scheduler,
)

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy

from opacus.accountants.utils import get_noise_multiplier
from transformers.trainer_pt_utils import DistributedSamplerWithLoop

from transformers.training_args import ParallelMode
from transformers.trainer_utils import has_length
from opacus.data_loader import DPDataLoader
from transformers.trainer_utils import seed_worker
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# this is tmd the code to run.
if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def dpzero_clip(loss_diff, C=1.0):
    tmp = torch.min(
        torch.ones_like(loss_diff),
        torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)),
    )
    return torch.mul(tmp, loss_diff)


class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, 
                 eval_dataset=None, tokenizer=None, model_init=None, compute_metrics=None,
                 callbacks=None, optimizers=(None, None), task_name=None, **kwargs):
        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init, 
                          compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers)
        self.task_name = task_name

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if "encoder.layer" in n:
                        try:
                            layer_num = int(
                                n[n.find("encoder.layer") + 14 :].split(".")[0]
                            )
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print("yes", n)
                            params[n] = p
                        else:
                            print("no ", n)
                    elif "embeddings" in n:
                        print("no ", n)
                    else:
                        print("yes", n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in params.items()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in params.items() if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == "adam":
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == "sgd":
                self.optimizer = SGD(
                    optimizer_grouped_parameters, lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (
            not self.args.layer_wise_optim
            or f".{self.state.global_step % self.model.config.num_hidden_layers}."
            in name
        ) and param.requires_grad

    def zo_forward(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.state.zo_forward_step += 1
        return loss.detach()
    
    def compute_model_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach()
    
    def efficient_perturb_parameters(
        self, model: nn.Module, random_seed: int, scaling_factor=1
    ):
        scalar = scaling_factor * self.args.zero_order_eps   
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0,std=1,size=param.data.size(),device=param.data.device,dtype=param.data.dtype)
            param.data = param.data + scalar * z 
            del z
        return model

    def perturb_parameters(
        self, model: nn.Module, random_vector=None, scaling_factor=1
    ):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0,std=1,size=param.data.size(),device=param.data.device,dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + (scaling_factor * self.args.zero_order_eps) * z 
            del z
        return model, random_vector

    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(
                1,
                int(
                    self.state.global_step
                    / self.args.max_steps
                    * self.args.zero_order_sample
                ),
            )
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    def get_train_dataloader(self) -> DataLoader:
        # combine poisson sampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # Use the sampler_seed if provided, otherwise fall back to data_seed or args.seed
            if hasattr(self.args, "sampler_seed") and self.args.sampler_seed is not None:
                seed = self.args.sampler_seed
                print(f"Using specific sampler_seed for batch sampling: {seed}")
            elif self.args.data_seed is not None:
                seed = self.args.data_seed
            else:
                seed = self.args.seed
            generator.manual_seed(seed)

        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )

        if self.args.world_size <= 1:
            train_batch_sampler = UniformWithReplacementSampler(
                num_samples=len(train_dataset),  # type: ignore[assignment, arg-type]
                sample_rate=self.args.dp_sample_rate,
                generator=generator,
            )
        else:
            train_batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(train_dataset),  # type: ignore[assignment, arg-type]
                sample_rate=self.args.dp_sample_rate,
                generator=generator,
            )

        return DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def log_batch_identifiers(self, inputs, epoch, step, num_batches_to_log=3):
        """
        Log unique identifiers for examples in a batch to verify batch reproducibility.
        Only logs for the first few batches in each epoch.
        
        Args:
            inputs: Batch inputs
            epoch: Current epoch number
            step: Current step/batch number
            num_batches_to_log: Number of batches to log at the beginning of each epoch
        """
        if not hasattr(inputs, "input_ids") or step >= num_batches_to_log:
            return
            
        if isinstance(inputs.input_ids, torch.Tensor):
            batch_identifier = []
            if inputs.input_ids.dim() > 1:
                # Get a unique identifier for each example in batch
                for i in range(min(5, inputs.input_ids.size(0))):
                    # Create a better identifier using more tokens and a hash
                    example_ids = inputs.input_ids[i].cpu().numpy()
                    example_hash = hash(example_ids.tobytes())
                    # Include first 3 tokens, last 3 tokens, and hash
                    first_tokens = ','.join([str(t) for t in example_ids[:3]])
                    last_tokens = ','.join([str(t) for t in example_ids[-3:]])
                    identifier = f"[{first_tokens}...{last_tokens}]#{example_hash % 10000}"
                    batch_identifier.append(identifier)
            
            logger.info(f"Epoch {epoch}, Batch {step}, sampler_seed={self.args.sampler_seed if hasattr(self.args, 'sampler_seed') else 'None'}")
            logger.info(f"  Batch identifiers: {batch_identifier}")
        else:
            logger.info(f"Epoch {epoch}, Batch {step}: Input type: {type(inputs.input_ids)}")

                
    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(
                model_path, dev_objective
            )  # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = (
            dev_objective if dev_objective is not None else default_dev_objective
        )

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        
        # Log batch sampler indices for the first few batches to verify reproducibility
        if hasattr(train_dataloader, 'batch_sampler') and hasattr(self.args, 'sampler_seed') and self.args.sampler_seed is not None:
            logger.info(f"Sampling with seed {self.args.sampler_seed}")
            batch_count = min(3, len(train_dataloader.batch_sampler))  # Log at most 3 batches
            for i in range(batch_count):
                # Get batch indices for the i-th batch
                if hasattr(train_dataloader.batch_sampler, '_indices'):
                    # For UniformWithReplacementSampler
                    indices = train_dataloader.batch_sampler._indices[i*train_dataloader.batch_sampler.batch_size:(i+1)*train_dataloader.batch_sampler.batch_size]
                    logger.info(f"Batch {i} indices (first 10): {indices[:10]}")
                else:
                    logger.info(f"Batch sampler type: {type(train_dataloader.batch_sampler)}")
        
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.args.gradient_accumulation_steps
        )
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(model_path, "optimizer.pt"),
                    map_location=self.args.device,
                )
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(model_path, "scheduler.pt"))
            )

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=self.args.fp16_opt_level
            )

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Total optimization steps = %d", t_total)

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info(
                    "  Continuing training from global step %d", self.state.global_step
                )
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # set up dp parameters:
        if self.args.dpzero:
            if self.args.dp_epsilon > 0:
                multiplier = get_noise_multiplier(
                    target_epsilon=self.args.dp_epsilon* 0.95,  # New: give some budget for N
                    target_delta=self.args.dp_delta,
                    steps=t_total,
                    sample_rate=self.args.dp_sample_rate,
                )

                b = 1 / (self.args.dp_epsilon * 0.05)
                # Use derived seed for Laplace noise if specified
                if self.args.random_direction_seed != -1:
                    # Set PyTorch's random seed for reproducible Laplace noise
                    # Use a combination of the base seed and the initialization step (0)
                    # This ensures the Laplace noise is the same across runs but different
                    # from other random noise in the training process
                    derived_seed = self.args.random_direction_seed + 0  # use 0 for initialization
                    torch.manual_seed(derived_seed)
                    # print(f"Using derived random seed for Laplace noise: {derived_seed} (base: {self.args.random_direction_seed}, init step: 0)")
                # Sample random noise from Laplace distribution
                laplace_noise = torch.distributions.Laplace(loc=0, scale=b).sample()
                noisy_batch_size = (
                    len(self.train_dataset) + laplace_noise
                ) * self.args.dp_sample_rate
                print(f'noisy expected batch size {noisy_batch_size}')
                
                self.dpzero_gaussian_std = (
                    multiplier * self.args.dpzero_clip_threshold / noisy_batch_size
                )
                print(f'L2-norm clipping for grads {self.args.dpzero_clip_threshold}')
                print(f'Std of Gaussian to add to grad``mean\'\' {self.dpzero_gaussian_std}')
            else:
                print(f'Non-DP !')
            
        for epoch in range(epochs_trained, int(num_train_epochs)):

            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):
                # self.log_batch_identifiers(inputs, epoch, step)
                
                if self.args.sync_embedding_layers:
                    assert (
                        model.module.model_type == "opt"
                    ), "did not implement embedding layer synchronization for non-OPT models"
                    model.module.model.decoder.embed_tokens.weight = (
                        model.module.lm_head.weight
                    )

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    total_params = 0
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))
                            total_params += 1
                    # print(f"Total number of parameters to optimize: {total_params}")

                    grads = []
                    # Use fixed seed for random projections if specified
                    if self.args.random_direction_seed != -1:
                        # Set NumPy's random seed for reproducible random directions
                        # Use a combination of the base seed and the current step
                        derived_seed = self.args.random_direction_seed + self.state.global_step
                        np.random.seed(derived_seed)
                        # print(f"Using derived random seed for directions: {derived_seed} (base: {self.args.random_direction_seed}, step: {self.state.global_step})")
                    all_random_seeds = np.random.randint(
                        1000000000, size=self.args.n)
                    # print(f"All random seeds: {all_random_seeds}")
                    # update_start_time = time.time()
                    for multi_d_idx in range(self.args.n):
                        # print("DPZero multi-d: ", multi_d_idx)
                        random_seed = all_random_seeds[multi_d_idx]
                        with torch.no_grad():
                            model = self.efficient_perturb_parameters(
                                model, random_seed)
                            loss1 = self.zo_forward(model, inputs)
                            model = self.efficient_perturb_parameters(
                                model, random_seed, scaling_factor=-2)
                            loss2 = self.zo_forward(model, inputs)
                        
                        projected_grad = (loss1 - loss2) / (
                            2 * self.args.zero_order_eps
                        )
                        grads.append(projected_grad)
                        model = self.efficient_perturb_parameters(model, random_seed)
                    # Stack 
                    stacked_grads = torch.stack(grads, dim=0) / self.args.n
                    
                    # clip
                    if self.args.dp_epsilon > 0:
                        norms = torch.norm(stacked_grads, dim=0, keepdim=True)
                        scaling_factors = self.args.dpzero_clip_threshold / norms
                        scaling_factors = torch.clamp(scaling_factors, max=1.0)
                        # count clipping # happen
                    else:
                        scaling_factors = 1           
                    clipped_grads = stacked_grads * scaling_factors
                    
                    # avg and add noise
                    if self.args.dp_epsilon > 0:
                        mean_grad = clipped_grads.sum(dim=1) / noisy_batch_size
                        # Use a derived seed for Gaussian noise if specified
                        if self.args.random_direction_seed != -1:
                            derived_seed = self.args.random_direction_seed + self.state.global_step
                            torch.manual_seed(derived_seed)
                        gaussian_noise = torch.randn(torch.tensor(self.args.n), 
                                                 device=self.args.device) * (self.dpzero_gaussian_std)
                        noisy_mean_grad = mean_grad + gaussian_noise
                    else:
                        noisy_mean_grad = clipped_grads.sum(dim=1) / stacked_grads.shape[1]
                    
                    # update_start_time = time.time()
                    for j in range(self.args.n):
                        random_seed = all_random_seeds[j]
                        scalar = self.args.learning_rate * noisy_mean_grad[j]
                        torch.manual_seed(random_seed)
                        for name, param in self.named_parameters_to_optim:
                            z = torch.normal(mean=0,std=1,size=param.data.size(),device=param.data.device,dtype=param.data.dtype)
                            param.data = param.data - scalar * z
                            del z
                    
                    del clipped_grads
                    del noisy_mean_grad
                    torch.cuda.empty_cache()
                    
                    logs = {}
                    with torch.no_grad():
                        eval_loss = self.compute_model_loss(model, inputs)
                    logs["loss"] = eval_loss.mean().item()
                    logs["loss"] = round(logs["loss"], 4)
                    logs["cur_step"] = self.state.global_step
                    logs["max_steps"] = self.args.max_steps
                    logs["clip"] = self.args.dpzero_clip_threshold
                    logs["lr"] = self.args.learning_rate
                    logs["n"] = self.args.n
                    logs["dp_epsilon"] = self.args.dp_epsilon
                    logs["sec"] = int(time.time() - start_time)
                    logs["min"] = (time.time() - start_time) / 60.0
                    logs["min"] = round(logs["min"], 2)
                    self.log(logs)
                    logger.info(str(logs))
                    torch.cuda.empty_cache()

                    if self.state.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                if self.state.global_step >= self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.state.global_step >= self.args.max_steps:
                break

            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        self.evaluate()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        return (
            TrainOutput(
                self.state.global_step, tr_loss / self.state.global_step, metrics
            ),
            self.objective,
        )

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
