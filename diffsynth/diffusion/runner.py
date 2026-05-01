import os
import random
import torch
from tqdm import tqdm
from accelerate import Accelerator
from ..core import load_state_dict
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

# 从这里到306行是为了支持续训所增加的
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


def _compute_epoch_resume(micro_steps_per_epoch, num_epochs, resume_step, max_train_steps, grad_accum_steps):
    if micro_steps_per_epoch <= 0:
        raise ValueError("Dataloader is empty. Cannot start training.")
    if grad_accum_steps <= 0:
        raise ValueError(f"`gradient_accumulation_steps` must be > 0, got {grad_accum_steps}.")
    optimizer_steps_per_epoch = (micro_steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps
    start_epoch = resume_step // optimizer_steps_per_epoch
    skip_optimizer_steps_in_epoch = resume_step % optimizer_steps_per_epoch
    skip_micro_steps_in_epoch = min(skip_optimizer_steps_in_epoch * grad_accum_steps, micro_steps_per_epoch)
    effective_num_epochs = num_epochs
    if max_train_steps is not None:
        # Ensure enough epochs to reach max_train_steps even when resuming from a later step.
        min_epochs = (max_train_steps + optimizer_steps_per_epoch - 1) // optimizer_steps_per_epoch
        effective_num_epochs = max(num_epochs, min_epochs)
    return start_epoch, skip_micro_steps_in_epoch, optimizer_steps_per_epoch, effective_num_epochs


def _state_dir(output_path, step):
    return os.path.join(output_path, "training_state", f"step-{step}")


def _legacy_state_dir(output_path, step):
    return os.path.join(output_path, f"accelerate_state_step-{step}")


def _lightweight_state_file(state_dir):
    return os.path.join(state_dir, "training_state.pt")


def _lightweight_rng_file(state_dir, rank):
    return os.path.join(state_dir, f"rng_state_rank{rank}.pt")


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None
    return state


def _restore_rng_state(state):
    if state is None:
        return
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "numpy" in state and state["numpy"] is not None:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _torch_load_pickle_compatible(path, map_location="cpu"):
    try:
        # PyTorch 2.6 defaults to weights_only=True, which cannot load Python/Numpy RNG objects.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without `weights_only` arg.
        return torch.load(path, map_location=map_location)


def _resolve_module_by_path(root_module, module_path):
    module = root_module
    for name in module_path.split("."):
        if not hasattr(module, name):
            return None
        module = getattr(module, name)
    return module


def _resolve_resume_checkpoint_path(model_logger: ModelLogger, resume_step: int, optimizer_steps_per_epoch: int):
    step_checkpoint = os.path.join(model_logger.checkpoint_path, f"step-{resume_step}.safetensors")
    if os.path.isfile(step_checkpoint):
        return step_checkpoint
    if optimizer_steps_per_epoch > 0 and resume_step > 0 and resume_step % optimizer_steps_per_epoch == 0:
        epoch_id = resume_step // optimizer_steps_per_epoch - 1
        epoch_checkpoint = os.path.join(model_logger.checkpoint_path, f"epoch-{epoch_id}.safetensors")
        if os.path.isfile(epoch_checkpoint):
            return epoch_checkpoint
    return None


def _load_resume_checkpoint(
    accelerator: Accelerator,
    model_logger: ModelLogger,
    model: DiffusionTrainingModule,
    resume_step: int,
    optimizer_steps_per_epoch: int,
):
    checkpoint_path = _resolve_resume_checkpoint_path(model_logger, resume_step, optimizer_steps_per_epoch)
    if checkpoint_path is None:
        step_checkpoint = os.path.join(model_logger.checkpoint_path, f"step-{resume_step}.safetensors")
        hint = [step_checkpoint]
        if optimizer_steps_per_epoch > 0 and resume_step > 0 and resume_step % optimizer_steps_per_epoch == 0:
            epoch_id = resume_step // optimizer_steps_per_epoch - 1
            hint.append(os.path.join(model_logger.checkpoint_path, f"epoch-{epoch_id}.safetensors"))
        raise FileNotFoundError(
            f"Trainable checkpoint for resume_step={resume_step} is missing. Expected one of: {hint}. "
            f"Please keep `output_path/checkpoints` when using lightweight resume."
        )

    state_dict = load_state_dict(checkpoint_path, torch_dtype=None, device="cpu")
    unwrapped_model = accelerator.unwrap_model(model)
    target_module = unwrapped_model
    remove_prefix = getattr(model_logger, "remove_prefix_in_ckpt", None)
    if isinstance(remove_prefix, str) and len(remove_prefix) > 0:
        module_path = remove_prefix[:-1] if remove_prefix.endswith(".") else remove_prefix
        maybe_target = _resolve_module_by_path(unwrapped_model, module_path)
        if maybe_target is not None:
            target_module = maybe_target

    if any(("lora_A.weight" in key or "lora_B.weight" in key) for key in state_dict.keys()):
        if hasattr(unwrapped_model, "mapping_lora_state_dict"):
            state_dict = unwrapped_model.mapping_lora_state_dict(state_dict)

    load_result = target_module.load_state_dict(state_dict, strict=False)
    if accelerator.is_main_process:
        print(f"Resume checkpoint loaded: {checkpoint_path}, keys={len(state_dict)}")
        if len(load_result.unexpected_keys) > 0:
            print(
                f"Warning: {len(load_result.unexpected_keys)} unexpected keys while loading resume checkpoint. "
                f"Example keys: {load_result.unexpected_keys[:5]}"
            )


def _load_accelerate_state_if_available(
    accelerator: Accelerator,
    model_logger: ModelLogger,
    model: DiffusionTrainingModule,
    optimizer,
    scheduler,
    resume_step: int,
    optimizer_steps_per_epoch: int,
):
    if resume_step <= 0:
        return
    accelerator.wait_for_everyone()
    state_dir = _state_dir(model_logger.output_path, resume_step)
    legacy_state_dir = _legacy_state_dir(model_logger.output_path, resume_step)

    lightweight_state_file = _lightweight_state_file(state_dir)
    if os.path.isfile(lightweight_state_file):
        _load_resume_checkpoint(accelerator, model_logger, model, resume_step, optimizer_steps_per_epoch)
        state = _torch_load_pickle_compatible(lightweight_state_file, map_location="cpu")
        saved_grad_accum_steps = state.get("gradient_accumulation_steps")
        current_grad_accum_steps = int(getattr(accelerator, "gradient_accumulation_steps", 1))
        if saved_grad_accum_steps is not None and int(saved_grad_accum_steps) != current_grad_accum_steps:
            raise ValueError(
                f"gradient_accumulation_steps mismatch: checkpoint={saved_grad_accum_steps}, "
                f"current={current_grad_accum_steps}. Please resume with the same accumulation steps."
            )
        step_unit = state.get("step_unit", None)
        if step_unit is None and accelerator.is_main_process:
            print(
                "Warning: loaded legacy lightweight state without step-unit metadata. "
                "Ensure `resume_step` matches the checkpoint naming used in that run."
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler_state = state.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = state.get("scaler")
        if scaler_state is not None and getattr(accelerator, "scaler", None) is not None:
            accelerator.scaler.load_state_dict(scaler_state)
        rng_file = _lightweight_rng_file(state_dir, accelerator.process_index)
        if not os.path.isfile(rng_file):
            rng_file = _lightweight_rng_file(state_dir, 0)
        if os.path.isfile(rng_file):
            _restore_rng_state(_torch_load_pickle_compatible(rng_file, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"Lightweight training state loaded from: {state_dir}")
        accelerator.wait_for_everyone()
    elif os.path.isdir(state_dir):
        accelerator.load_state(state_dir)
        if accelerator.is_main_process:
            print(f"Accelerate full state loaded from: {state_dir}")
    elif os.path.isdir(legacy_state_dir):
        accelerator.load_state(legacy_state_dir)
        if accelerator.is_main_process:
            print(f"Accelerate full state loaded from legacy path: {legacy_state_dir}")
    else:
        raise FileNotFoundError(
            f"Training state not found for resume_step={resume_step}: "
            f"{state_dir} (or legacy path {legacy_state_dir}). "
            f"Please resume from a saved step where `{os.path.basename(state_dir)}` exists "
            f"under output_path."
        )


def _save_accelerate_state(accelerator: Accelerator, model_logger: ModelLogger, optimizer, scheduler, step: int):
    state_dir = _state_dir(model_logger.output_path, step)
    accelerator.wait_for_everyone()
    os.makedirs(state_dir, exist_ok=True)
    if accelerator.is_main_process:
        state = {
            "step": int(step),
            "step_unit": "optimizer_step_after_accumulation",
            "gradient_accumulation_steps": int(getattr(accelerator, "gradient_accumulation_steps", 1)),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": accelerator.scaler.state_dict() if getattr(accelerator, "scaler", None) is not None else None,
        }
        torch.save(state, _lightweight_state_file(state_dir))
    torch.save(_capture_rng_state(), _lightweight_rng_file(state_dir, accelerator.process_index))
    accelerator.wait_for_everyone()


def _infer_per_device_batch_size(dataloader):
    batch_size = getattr(dataloader, "batch_size", None)
    if batch_size is None:
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        batch_size = getattr(batch_sampler, "batch_size", None)
    if batch_size is None:
        # This project uses default DataLoader batch_size=1 in training.
        batch_size = 1
    return int(batch_size)


def _count_model_parameters(model: torch.nn.Module):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        num_params = int(param.numel())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return total_params, trainable_params


def _print_training_plan(
    accelerator: Accelerator,
    optimizer_steps_per_epoch: int,
    effective_num_epochs: int,
    resume_step: int,
    max_train_steps: int,
    grad_accum_steps: int,
    per_device_batch_size: int,
    total_params: int,
    trainable_params: int,
):
    if not accelerator.is_main_process:
        return
    world_size = int(getattr(accelerator, "num_processes", 1))
    global_micro_batch_size = per_device_batch_size * world_size
    global_batch_size_with_accum = global_micro_batch_size * grad_accum_steps
    total_steps = max_train_steps if max_train_steps is not None else effective_num_epochs * optimizer_steps_per_epoch
    remaining_steps = max(total_steps - resume_step, 0)
    print("*" * 60)
    print("=== Training Plan ===")
    print(f"epochs={effective_num_epochs}")
    print(f"total_steps={total_steps}")
    print(f"remaining_steps_from_resume={remaining_steps}")
    print(f"steps_per_epoch_after_accumulation={optimizer_steps_per_epoch}")
    print(f"global_batch_size_after_accumulation={global_batch_size_with_accum}")
    print(f"gradient_accumulation_steps={grad_accum_steps}")
    print(f"world_size={world_size}")
    print(f"per_device_micro_batch_size={per_device_batch_size}")
    print(f"global_micro_batch_size={global_micro_batch_size}")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")
    print("*" * 60)


def _append_accumulated_custom_metrics_csv(model_logger: ModelLogger, step: int, epoch_id: int, global_count: int, metric_means: dict):
    os.makedirs(model_logger.log_path, exist_ok=True)
    metrics_path = os.path.join(model_logger.log_path, "maskdpo_metrics.csv")
    ordered_names = ["win_diff_reward", "lose_diff_reward", "reward_margin", "reward_acc", "loss"]
    names = [name for name in ordered_names if name in metric_means]
    names += [name for name in metric_means.keys() if name not in names]

    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        header = ["step", "epoch", "global_batch_size"] + names
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    row_values = [str(int(step)), str(int(epoch_id)), str(int(global_count))]
    row_values += [f"{float(metric_means[name]):.8f}" for name in names]
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(",".join(row_values) + "\n")


def _print_accumulated_custom_metrics(
    accelerator: Accelerator,
    model: torch.nn.Module,
    model_logger: ModelLogger,
    step: int,
    epoch_id: int = -1,
):
    model_unwrapped = accelerator.unwrap_model(model)
    pop_metrics_fn = getattr(model_unwrapped, "pop_accumulated_log_metrics", None)
    if not callable(pop_metrics_fn):
        return
    payload = pop_metrics_fn()
    if payload is None:
        return

    metric_sums = payload.get("metric_sums", {})
    sample_count = int(payload.get("sample_count", 0))
    if sample_count <= 0 or len(metric_sums) == 0:
        return

    count_tensor = torch.tensor([float(sample_count)], device=accelerator.device, dtype=torch.float32)
    global_count = accelerator.gather(count_tensor).sum().item()
    if global_count <= 0:
        return

    metric_means = {}
    for name, local_sum in metric_sums.items():
        if isinstance(local_sum, torch.Tensor):
            local_sum_tensor = local_sum.detach().float().reshape(1)
        else:
            local_sum_tensor = torch.tensor([float(local_sum)], device=accelerator.device, dtype=torch.float32)
        if local_sum_tensor.device != accelerator.device:
            local_sum_tensor = local_sum_tensor.to(accelerator.device)
        global_sum = accelerator.gather(local_sum_tensor).sum().item()
        metric_means[name] = global_sum / global_count

    if accelerator.is_main_process:
        _append_accumulated_custom_metrics_csv(
            model_logger=model_logger,
            step=step,
            epoch_id=epoch_id,
            global_count=int(global_count),
            metric_means=metric_means,
        )
        ordered_names = ["win_diff_reward", "lose_diff_reward", "reward_margin", "reward_acc", "loss"]
        names = [name for name in ordered_names if name in metric_means]
        names += [name for name in metric_means.keys() if name not in names]
        message = ", ".join([f"{name}={metric_means[name]:.8f}" for name in names])
        print(f"[metrics][step={step}] {message}")



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
    total_params, trainable_params = _count_model_parameters(model)
    grad_accum_steps = int(getattr(accelerator, "gradient_accumulation_steps", 1))

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    micro_steps_per_epoch = len(dataloader)
    start_epoch, skip_micro_steps_in_epoch, optimizer_steps_per_epoch, effective_num_epochs = _compute_epoch_resume(
        micro_steps_per_epoch, num_epochs, resume_step, max_train_steps, grad_accum_steps
    )
    _load_accelerate_state_if_available(
        accelerator, model_logger, model, optimizer, scheduler, resume_step, optimizer_steps_per_epoch
    )
    _print_training_plan(
        accelerator,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        effective_num_epochs=effective_num_epochs,
        resume_step=resume_step,
        max_train_steps=max_train_steps,
        grad_accum_steps=grad_accum_steps,
        per_device_batch_size=_infer_per_device_batch_size(dataloader),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    model_logger.num_steps = resume_step
    if max_train_steps is not None and resume_step >= max_train_steps:
        print(f"resume_step ({resume_step}) >= max_train_steps ({max_train_steps}), nothing to train.")
        return
    if start_epoch >= effective_num_epochs:
        print(
            f"resume_step ({resume_step}) is beyond available epochs "
            f"(num_epochs={effective_num_epochs}, steps_per_epoch_after_accumulation={optimizer_steps_per_epoch}). Nothing to train."
        )
        return

    reached_max_train_steps = False
    for epoch_id in range(start_epoch, effective_num_epochs):
        for micro_step_in_epoch, data in enumerate(tqdm(dataloader)):
            if epoch_id == start_epoch and micro_step_in_epoch < skip_micro_steps_in_epoch:
                continue
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id)
                    _print_accumulated_custom_metrics(
                        accelerator,
                        model,
                        model_logger,
                        model_logger.num_steps,
                        epoch_id=epoch_id,
                    )
                    if save_steps is not None and model_logger.num_steps % save_steps == 0:
                        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
                scheduler.step()
                optimizer.zero_grad()
            if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                reached_max_train_steps = True
                break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
        if reached_max_train_steps:
            break

    if reached_max_train_steps and save_steps is None:
        # Ensure the exact final step is checkpointed when stopping mid-epoch by max_train_steps.
        model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")
        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
    model_logger.on_training_end(accelerator, model, save_steps)
    if save_steps is not None and model_logger.num_steps % save_steps != 0:
        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)


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
    total_params, trainable_params = _count_model_parameters(model)
    grad_accum_steps = int(getattr(accelerator, "gradient_accumulation_steps", 1))

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    micro_steps_per_epoch = len(dataloader)
    start_epoch, skip_micro_steps_in_epoch, optimizer_steps_per_epoch, effective_num_epochs = _compute_epoch_resume(
        micro_steps_per_epoch, num_epochs, resume_step, max_train_steps, grad_accum_steps
    )
    _load_accelerate_state_if_available(
        accelerator, model_logger, model, optimizer, scheduler, resume_step, optimizer_steps_per_epoch
    )
    _print_training_plan(
        accelerator,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        effective_num_epochs=effective_num_epochs,
        resume_step=resume_step,
        max_train_steps=max_train_steps,
        grad_accum_steps=grad_accum_steps,
        per_device_batch_size=_infer_per_device_batch_size(dataloader),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    model_logger.num_steps = resume_step
    if max_train_steps is not None and resume_step >= max_train_steps:
        print(f"resume_step ({resume_step}) >= max_train_steps ({max_train_steps}), nothing to train.")
        return
    if start_epoch >= effective_num_epochs:
        print(
            f"resume_step ({resume_step}) is beyond available epochs "
            f"(num_epochs={effective_num_epochs}, steps_per_epoch_after_accumulation={optimizer_steps_per_epoch}). Nothing to train."
        )
        return

    reached_max_train_steps = False
    for epoch_id in range(start_epoch, effective_num_epochs):
        for micro_step_in_epoch, data in enumerate(tqdm(dataloader)):
            if epoch_id == start_epoch and micro_step_in_epoch < skip_micro_steps_in_epoch:
                continue
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss, epoch_id=epoch_id)
                    _print_accumulated_custom_metrics(
                        accelerator,
                        model,
                        model_logger,
                        model_logger.num_steps,
                        epoch_id=epoch_id,
                    )
                    if save_steps is not None and model_logger.num_steps % save_steps == 0:
                        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
                scheduler.step()
                optimizer.zero_grad()
            if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
                reached_max_train_steps = True
                break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
        if reached_max_train_steps:
            break

    if reached_max_train_steps and save_steps is None:
        model_logger.save_model(accelerator, model, f"step-{model_logger.num_steps}.safetensors")
        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)
    model_logger.on_training_end(accelerator, model, save_steps)
    if save_steps is not None and model_logger.num_steps % save_steps != 0:
        _save_accelerate_state(accelerator, model_logger, optimizer, scheduler, model_logger.num_steps)


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
