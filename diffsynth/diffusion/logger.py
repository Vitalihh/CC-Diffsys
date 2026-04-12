import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x:x,
        preview_steps=0,
        preview_kwargs=None,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.loss_log_path = os.path.join(self.output_path, "loss.csv")
        self.loss_log_initialized = False
        self.preview_steps = max(0, int(preview_steps))
        self.preview_kwargs = {} if preview_kwargs is None else preview_kwargs
        self.preview_warning_emitted = False

    def _reduce_loss(self, accelerator: Accelerator, loss):
        if isinstance(loss, torch.Tensor):
            loss_tensor = loss.detach().float()
            if loss_tensor.ndim == 0:
                loss_tensor = loss_tensor.reshape(1)
            else:
                loss_tensor = loss_tensor.reshape(-1).mean().reshape(1)
        else:
            loss_tensor = torch.tensor([float(loss)], device=accelerator.device, dtype=torch.float32)
        if loss_tensor.device != accelerator.device:
            loss_tensor = loss_tensor.to(accelerator.device)
        reduced_loss = accelerator.gather(loss_tensor).mean().item()
        return reduced_loss

    def _append_loss(self, epoch_id, loss_value):
        os.makedirs(self.output_path, exist_ok=True)
        if not self.loss_log_initialized:
            if not os.path.exists(self.loss_log_path) or os.path.getsize(self.loss_log_path) == 0:
                with open(self.loss_log_path, "w", encoding="utf-8") as f:
                    f.write("step,epoch,loss\n")
            self.loss_log_initialized = True
        with open(self.loss_log_path, "a", encoding="utf-8") as f:
            f.write(f"{self.num_steps},{epoch_id},{loss_value:.8f}\n")

    def _run_preview(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        if self.preview_steps <= 0 or self.num_steps % self.preview_steps != 0:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            model_unwrapped = accelerator.unwrap_model(model)
            preview_fn = getattr(model_unwrapped, "generate_preview", None)
            if callable(preview_fn):
                try:
                    preview_fn(
                        output_path=self.output_path,
                        step=self.num_steps,
                        epoch_id=epoch_id,
                        **self.preview_kwargs,
                    )
                except Exception as e:
                    print(f"Preview generation failed at step {self.num_steps}: {e}")
            elif not self.preview_warning_emitted:
                print("Preview generation is enabled but `generate_preview` is not implemented in this model. Skipping previews.")
                self.preview_warning_emitted = True
        accelerator.wait_for_everyone()


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        loss = kwargs.get("loss")
        epoch_id = kwargs.get("epoch_id", -1)
        if loss is not None:
            loss_value = self._reduce_loss(accelerator, loss)
            if accelerator.is_main_process:
                self._append_loss(epoch_id, loss_value)
        self._run_preview(accelerator, model, epoch_id)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
