import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _parse_step_control_args(args, resume_step, max_train_steps):
    if args is not None:
        if hasattr(args, "resume_step"):
            resume_step = args.resume_step
        if hasattr(args, "max_train_steps"):
            max_train_steps = args.max_train_steps
    resume_step = 0 if resume_step is None else int(resume_step)
    if resume_step < 0:
        raise ValueError(f"`resume_step` must be >= 0, got {resume_step}.")
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            raise ValueError(f"`max_train_steps` must be > 0, got {max_train_steps}.")
    return resume_step, max_train_steps


def _compute_epoch_resume(steps_per_epoch, num_epochs, resume_step, max_train_steps):
    if steps_per_epoch <= 0:
        raise ValueError("Dataloader is empty. Cannot start training.")
    start_epoch = resume_step // steps_per_epoch
    skip_steps_in_epoch = resume_step % steps_per_epoch
    effective_num_epochs = num_epochs
    if max_train_steps is not None:
        # Ensure enough epochs to reach max_train_steps even when resuming from a later step.
        min_epochs = (max_train_steps + steps_per_epoch - 1) // steps_per_epoch
        effective_num_epochs = max(num_epochs, min_epochs)
    return start_epoch, skip_steps_in_epoch, effective_num_epochs


def _state_dir(output_path, step):
    return os.path.join(output_path, f"accelerate_state_step-{step}")


def _load_accelerate_state_if_available(accelerator: Accelerator, model_logger: ModelLogger, resume_step: int):
    if resume_step <= 0:
        return
    state_dir = _state_dir(model_logger.output_path, resume_step)
    if os.path.isdir(state_dir):
        accelerator.load_state(state_dir)
        print(f"Accelerate state loaded from: {state_dir}")
    else:
        raise FileNotFoundError(
            f"Accelerate state not found for resume_step={resume_step}: {state_dir}. "
            f"Please resume from a saved step where `{os.path.basename(state_dir)}` exists "
            f"under output_path."
        )


def _save_accelerate_state(accelerator: Accelerator, model_logger: ModelLogger, step: int):
    state_dir = _state_dir(model_logger.output_path, step)
    accelerator.save_state(state_dir)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    resume_step: int = 0,
    max_train_steps: int = None,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    resume_step, max_train_steps = _parse_step_control_args(args, resume_step, max_train_steps)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    _load_accelerate_state_if_available(accelerator, model_logger, resume_step)

    steps_per_epoch = len(dataloader)
    start_epoch, skip_steps_in_epoch, effective_num_epochs = _compute_epoch_resume(
        steps_per_epoch, num_epochs, resume_step, max_train_steps
    )
    model_logger.num_steps = resume_step
    if max_train_steps is not None and resume_step >= max_train_steps:
        print(f"resume_step ({resume_step}) >= max_train_steps ({max_train_steps}), nothing to train.")
        return
    if start_epoch >= effective_num_epochs:
        print(
            f"resume_step ({resume_step}) is beyond available epochs "
            f"(num_epochs={effective_num_epochs}, steps_per_epoch={steps_per_epoch}). Nothing to train."
        )
        return

    reached_max_train_steps = False
    for epoch_id in range(start_epoch, effective_num_epochs):
        for step_in_epoch, data in enumerate(tqdm(dataloader)):
            if epoch_id == start_epoch and step_in_epoch < skip_steps_in_epoch:
                continue
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id)
                if save_steps is not None and model_logger.num_steps % save_steps == 0:
                    _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
                scheduler.step()
                optimizer.zero_grad()
            if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                reached_max_train_steps = True
                break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
        if reached_max_train_steps:
            break

    if reached_max_train_steps and save_steps is None:
        # Ensure the exact final step is checkpointed when stopping mid-epoch by max_train_steps.
        model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")
        _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
    model_logger.on_training_end(accelerator, model, save_steps)
    if save_steps is not None and model_logger.num_steps % save_steps != 0:
        _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def launch_dpo_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    max_grad_norm: float = 1.0,
    resume_step: int = 0,
    max_train_steps: int = None,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    resume_step, max_train_steps = _parse_step_control_args(args, resume_step, max_train_steps)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    _load_accelerate_state_if_available(accelerator, model_logger, resume_step)

    steps_per_epoch = len(dataloader)
    start_epoch, skip_steps_in_epoch, effective_num_epochs = _compute_epoch_resume(
        steps_per_epoch, num_epochs, resume_step, max_train_steps
    )
    model_logger.num_steps = resume_step
    if max_train_steps is not None and resume_step >= max_train_steps:
        print(f"resume_step ({resume_step}) >= max_train_steps ({max_train_steps}), nothing to train.")
        return
    if start_epoch >= effective_num_epochs:
        print(
            f"resume_step ({resume_step}) is beyond available epochs "
            f"(num_epochs={effective_num_epochs}, steps_per_epoch={steps_per_epoch}). Nothing to train."
        )
        return

    reached_max_train_steps = False
    for epoch_id in range(start_epoch, effective_num_epochs):
        for step_in_epoch, data in enumerate(tqdm(dataloader)):
            if epoch_id == start_epoch and step_in_epoch < skip_steps_in_epoch:
                continue
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id)
                if save_steps is not None and model_logger.num_steps % save_steps == 0:
                    _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
                scheduler.step()
                optimizer.zero_grad()
            if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                reached_max_train_steps = True
                break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
        if reached_max_train_steps:
            break

    if reached_max_train_steps and save_steps is None:
        model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")
        _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)
    model_logger.on_training_end(accelerator, model, save_steps)
    if save_steps is not None and model_logger.num_steps % save_steps != 0:
        _save_accelerate_state(accelerator, model_logger, model_logger.num_steps)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False),
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
