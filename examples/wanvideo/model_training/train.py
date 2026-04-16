import torch, os, argparse, accelerate, warnings, copy
import numpy as np
from PIL import Image
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.unified_dataset import DPOVideoDataset, MaskDPOVideoDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MASK_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
MASK_TENSOR_EXTS = (".pt", ".pth")


def _resolve_dataset_path(base_path, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_path, path)


def _normalize_mask_file_count(mask_files, num_frames):
    if len(mask_files) == 0:
        raise ValueError("No mask images found.")
    if num_frames is None:
        return mask_files
    if len(mask_files) == 1 and num_frames > 1:
        return mask_files * num_frames
    if len(mask_files) < num_frames:
        raise ValueError(
            f"Mask count ({len(mask_files)}) is smaller than video frame count ({num_frames}). "
            f"Please provide at least {num_frames} masks."
        )
    if len(mask_files) > num_frames:
        return mask_files[:num_frames]
    return mask_files


def _load_mask_from_files(mask_files, num_frames, height, width, device=None, dtype=None):
    mask_files = _normalize_mask_file_count(mask_files, num_frames)
    if height is None or width is None:
        with Image.open(mask_files[0]) as first_mask:
            first_mask = first_mask.convert("L")
            if height is None:
                height = first_mask.height
            if width is None:
                width = first_mask.width

    mask_frames = []
    for mask_file in mask_files:
        with Image.open(mask_file) as mask_pil:
            mask_pil = mask_pil.convert("L")
            mask_pil = mask_pil.resize((width, height), Image.NEAREST)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_frames.append(mask_np)

    mask = torch.from_numpy(np.stack(mask_frames, axis=0)).unsqueeze(0).unsqueeze(0)  # [1, 1, F, H, W]
    if dtype is not None:
        mask = mask.to(dtype=dtype)
    if device is not None:
        mask = mask.to(device=device)
    return mask


def _torch_load_pickle_compatible(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_mask_from_torch_file(mask_file, map_location="cpu"):
    mask_data = _torch_load_pickle_compatible(mask_file, map_location=map_location)
    if isinstance(mask_data, dict):
        if "mask" in mask_data:
            mask_data = mask_data["mask"]
        elif "input_mask" in mask_data:
            mask_data = mask_data["input_mask"]
        else:
            raise TypeError(
                f"Mask tensor file contains dict without `mask`/`input_mask`: {mask_file}, keys={list(mask_data.keys())[:5]}"
            )
    if isinstance(mask_data, np.ndarray):
        mask_data = torch.from_numpy(mask_data)
    if not isinstance(mask_data, torch.Tensor):
        raise TypeError(f"Unsupported mask tensor file content type: {type(mask_data)} in {mask_file}")
    return mask_data


def load_mask_from_folder(mask_folder_path, num_frames, height, width, device=None, dtype=None):
    if not os.path.isdir(mask_folder_path):
        raise ValueError(f"`mask_folder_path` does not exist or is not a directory: {mask_folder_path}")
    mask_files = sorted(
        [
            os.path.join(mask_folder_path, name)
            for name in os.listdir(mask_folder_path)
            if os.path.splitext(name)[1].lower() in MASK_IMAGE_EXTS
        ]
    )
    if len(mask_files) == 0:
        raise ValueError(f"No mask images found in folder: {mask_folder_path}")
    return _load_mask_from_files(mask_files, num_frames, height, width, device=device, dtype=dtype)


def build_mask_operator(base_path, num_frames, height, width, fallback_video_operator):
    def mask_operator(mask_data):
        if isinstance(mask_data, list):
            if len(mask_data) == 0:
                raise ValueError("Mask list is empty.")
            if not all(isinstance(path, str) for path in mask_data):
                raise TypeError("Mask list must contain file paths.")
            mask_files = [_resolve_dataset_path(base_path, path) for path in mask_data]
            return _load_mask_from_files(mask_files, num_frames, height, width)

        if isinstance(mask_data, str):
            mask_path = _resolve_dataset_path(base_path, mask_data)
            if os.path.isdir(mask_path):
                return load_mask_from_folder(mask_path, num_frames, height, width)
            file_ext = os.path.splitext(mask_path)[1].lower()
            if file_ext in MASK_IMAGE_EXTS:
                return _load_mask_from_files([mask_path], num_frames, height, width)
            if file_ext in MASK_TENSOR_EXTS:
                return _load_mask_from_torch_file(mask_path, map_location="cpu")
            return fallback_video_operator(mask_data)

        raise TypeError(f"Unsupported mask data type: {type(mask_data)}")

    return mask_operator


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_alpha=None, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dpo_beta=500.0,
        dpo_ref_model_path=None,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.dpo_ref_models = {}
        if (task.startswith("dpo") or task.startswith("maskdpo")) and not task.endswith(":data_process"):
            ref_dit = copy.deepcopy(self.pipe.dit)
            ref_dit.eval()
            ref_dit.requires_grad_(False)
            if dpo_ref_model_path is not None:
                ref_state_dict = load_state_dict(dpo_ref_model_path, torch_dtype=None, device="cpu")
                load_result = ref_dit.load_state_dict(ref_state_dict, strict=False)
                print(f"DPO ref model loaded: {dpo_ref_model_path}, keys={len(ref_state_dict)}")
                if len(load_result.unexpected_keys) > 0:
                    print(
                        f"Warning: {len(load_result.unexpected_keys)} unexpected keys while loading DPO ref model. "
                        f"Example keys: {load_result.unexpected_keys[:5]}"
                    )
            self.dpo_ref_models["dit"] = ref_dit
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=trainable_models,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank,
            lora_checkpoint=lora_checkpoint,
            lora_alpha=lora_alpha,
            preset_lora_path=preset_lora_path,
            preset_lora_model=preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # maskdpo not use task_to_loss
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "dpo:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "dpo": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchDPOLoss(
                pipe, ref_dit=self._get_dpo_ref_dit(), dpo_beta=self.dpo_beta, **inputs_shared, **inputs_posi
            ),
            "dpo:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchDPOLoss(
                pipe, ref_dit=self._get_dpo_ref_dit(), dpo_beta=self.dpo_beta, **inputs_shared, **inputs_posi
            ),
            "maskdpo": None,
            "maskdpo:train": None,
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.dpo_beta = dpo_beta
        self._preview_prompt_warning_emitted = False

    def _get_dpo_ref_dit(self):
        ref_dit = self.dpo_ref_models.get("dit")
        if ref_dit is None:
            raise RuntimeError("DPO reference model is not initialized.")
        return ref_dit

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        ref_dit = self.dpo_ref_models.get("dit")
        if ref_dit is not None:
            ref_dit.to(*args, **kwargs)
        else:
            raise RuntimeError("DPO reference model is not initialized.")
        return self

    def _snapshot_scheduler_state(self):
        state = {}
        scheduler = self.pipe.scheduler
        for name in ("sigmas", "timesteps", "linear_timesteps_weights", "training"):
            if hasattr(scheduler, name):
                value = getattr(scheduler, name)
                if isinstance(value, torch.Tensor):
                    state[name] = value.clone()
                else:
                    state[name] = value
        return state

    def _restore_scheduler_state(self, state):
        scheduler = self.pipe.scheduler
        for name, value in state.items():
            setattr(scheduler, name, value)

    def generate_preview(
        self,
        output_path,
        step,
        epoch_id=-1,
        prompt=None,
        negative_prompt="",
        num_inference_steps=8,
        num_frames=81,
        height=None,
        width=None,
        seed=0,
        cfg_scale=1.0,
        fps=15.0,
    ):
        if prompt is None or len(str(prompt).strip()) == 0:
            if not self._preview_prompt_warning_emitted:
                print("Preview generation skipped because `preview_prompt` is empty.")
                self._preview_prompt_warning_emitted = True
            return
        height = int(height) if height is not None else 480
        width = int(width) if width is not None else 832
        num_frames = int(num_frames)
        num_inference_steps = int(num_inference_steps)

        preview_folder = os.path.join(output_path, "preview")
        os.makedirs(preview_folder, exist_ok=True)
        file_name = f"step-{step}.mp4" if epoch_id is None or epoch_id < 0 else f"epoch-{epoch_id}_step-{step}.mp4"
        save_path = os.path.join(preview_folder, file_name)

        scheduler_state = self._snapshot_scheduler_state()
        module_training_states = {id(module): module.training for module in self.pipe.modules()}

        try:
            self.pipe.eval()
            with torch.no_grad():
                video = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    seed=seed,
                    height=height,
                    width=width,
                    cfg_scale=cfg_scale,
                    tiled=True,
                    progress_bar_cmd=lambda x: x,
                )
            save_video(video, save_path, fps=float(fps), quality=5)
            print(f"Preview video saved: {save_path}")
        finally:
            for module in self.pipe.modules():
                module.training = module_training_states[id(module)]
            self._restore_scheduler_state(scheduler_state)
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        if self.task.startswith("maskdpo"):
            return self._get_maskdpo_pipeline_inputs(data)
        if self.task.startswith("dpo"):
            return self._get_dpo_pipeline_inputs(data)
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def _get_dpo_pipeline_inputs(self, data):
        video_chosen = data["video_chosen"]
        video_rejected = data["video_rejected"]
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video_chosen,
            "input_video_rejected": video_rejected,
            "height": video_chosen[0].size[1],
            "width": video_chosen[0].size[0],
            "num_frames": len(video_chosen),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def _get_maskdpo_pipeline_inputs(self, data):
        video_chosen = data["video_chosen"]
        video_rejected = data["video_rejected"]
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video_chosen,
            "input_video_rejected": video_rejected,
            "input_mask": data["mask"],
            "input_prompt_sft": data["prompt_sft"],
            "input_video_sft": data["video_sft"],
            "input_prompt_vdpo": data["prompt_vdpo"],
            "input_video_vdpo_chosen": data["video_vdpo_chosen"],
            "input_video_vdpo_rejected": data["video_vdpo_rejected"],
            "input_strength": data["strength"],
            "height": video_chosen[0].size[1],
            "width": video_chosen[0].size[0],
            "num_frames": len(video_chosen),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if self.task.startswith("maskdpo") and not self.task.endswith(":data_process"):
            return self._forward_maskdpo(inputs)
        if self.task.startswith("dpo") and not self.task.endswith(":data_process"):
            return self._forward_dpo(inputs)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def _forward_dpo(self, inputs):
        inputs_shared, inputs_posi, inputs_nega = inputs
        input_video_rejected = inputs_shared.pop("input_video_rejected")

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        chosen_latents = inputs_shared["input_latents"]

        with torch.no_grad():
            video_rej_tensor = self.pipe.preprocess_video(input_video_rejected)
            video_rej_tensor = video_rej_tensor.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            self.pipe.load_models_to_device(("vae",))
            rejected_latents = self.pipe.vae.encode(
                video_rej_tensor, device=self.pipe.device
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        inputs_shared["input_latents_chosen"] = chosen_latents
        inputs_shared["input_latents_rejected"] = rejected_latents

        loss = self.task_to_loss[self.task](self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return loss

    def _process_mask(self, mask_data, latent_shape):
        """
        Convert mask to a tensor aligned with latent shape.
        mask_data: list[PIL.Image] or Tensor
        latent_shape: [C, F, H, W]
        return: [1, F_latent, H_latent, W_latent]
        """
        if isinstance(mask_data, torch.Tensor):
            mask_5d = mask_data
            if mask_5d.ndim == 3:      # [F, H, W]
                mask_5d = mask_5d.unsqueeze(0).unsqueeze(0)
            elif mask_5d.ndim == 4:    # [1, F, H, W]
                mask_5d = mask_5d.unsqueeze(0)
            elif mask_5d.ndim != 5:    # [1, 1, F, H, W]
                raise ValueError(f"Unsupported mask tensor shape: {tuple(mask_5d.shape)}")
            mask_5d = mask_5d.to(dtype=torch.float32)
        elif isinstance(mask_data, list):
            if len(mask_data) == 0:
                raise ValueError("Mask frame list is empty.")
            mask_tensors = []
            for frame in mask_data:
                gray = frame.convert("L")
                arr = np.array(gray, dtype=np.float32) / 255.0
                mask_tensors.append(torch.from_numpy(arr))
            mask_5d = torch.stack(mask_tensors, dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, F, H, W]
        else:
            raise TypeError(f"Unsupported mask type: {type(mask_data)}")

        
        mask_5d = mask_5d.clamp(0, 1)
        
        if len(latent_shape) == 5:
            _, C_latent, F_latent, H_latent, W_latent = latent_shape
        else:
            raise ValueError(f"Unsupported latent shape for mask resize: {tuple(latent_shape)}")
        
        mask_5d = torch.nn.functional.interpolate(
            mask_5d, size=(F_latent, H_latent, W_latent), mode="trilinear", align_corners=False
        )
        mask_5d = mask_5d.clamp(0, 1)
        if mask_5d.shape[0] != 1:
            raise ValueError(f"Mask batch size ({mask_5d.shape[0]}) does not match expected batch size (1).")
        if mask_5d.shape[1] == 1:
            mask = mask_5d.repeat(1, C_latent,1,1,1)
        else:
            if mask_5d.shape[1] != C_latent:
                raise ValueError(f"Mask channel count ({mask_5d.shape[1]}) does not match latent channels ({C_latent}).")
            mask = mask_5d
        # mask:[1 C F H W]
        return mask

    def _encode_prompt(self, prompt):
        with torch.no_grad():
            self.pipe.load_models_to_device(("text_encoder",))
            ids, mask = self.pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to(self.pipe.device)
            mask = mask.to(self.pipe.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.pipe.text_encoder(ids, mask)
            for i, v in enumerate(seq_lens):
                prompt_emb[:, v:] = 0
        return prompt_emb.to(dtype=self.pipe.torch_dtype)

    def _forward_maskdpo(self, inputs):
        inputs_shared, inputs_posi, inputs_nega = inputs

        input_video_rejected = inputs_shared.pop("input_video_rejected")
        input_mask = inputs_shared.pop("input_mask")
        input_prompt_sft = inputs_shared.pop("input_prompt_sft")
        input_video_sft = inputs_shared.pop("input_video_sft")
        input_prompt_vdpo = inputs_shared.pop("input_prompt_vdpo")
        input_video_vdpo_chosen = inputs_shared.pop("input_video_vdpo_chosen")
        input_video_vdpo_rejected = inputs_shared.pop("input_video_vdpo_rejected")
        strength = inputs_shared.pop("input_strength")


        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        chosen_latents = inputs_shared["input_latents"]


        with torch.no_grad():
            self.pipe.load_models_to_device(("vae",))

            def encode_video(video_frames):
                tensor = self.pipe.preprocess_video(video_frames)
                tensor = tensor.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                return self.pipe.vae.encode(
                    tensor, device=self.pipe.device
                ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

            rejected_latents = encode_video(input_video_rejected)
            sft_latents = encode_video(input_video_sft)
            vdpo_chosen_latents = encode_video(input_video_vdpo_chosen)
            vdpo_rejected_latents = encode_video(input_video_vdpo_rejected)


        sft_context = self._encode_prompt(input_prompt_sft)
        vdpo_context = self._encode_prompt(input_prompt_vdpo)

        mask_tensor = self._process_mask(input_mask, chosen_latents.shape)
        mask_tensor = mask_tensor.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        base_inputs = {k: v for k, v in inputs_shared.items() if k != "input_latents"}
        base_inputs.update(inputs_posi)

        maskdpo_inputs = dict(base_inputs)
        maskdpo_inputs["input_latents_chosen"] = chosen_latents
        maskdpo_inputs["input_latents_rejected"] = rejected_latents
        maskdpo_inputs["strength"] = strength
        mask_dpo_loss = FlowMatchMaskDPOLoss(
            self.pipe, ref_dit=self._get_dpo_ref_dit(),
            dpo_beta=self.dpo_beta, mask=mask_tensor, **maskdpo_inputs
        )

        sft_inputs = {k: v for k, v in base_inputs.items() if k != "context"}
        sft_inputs["context"] = sft_context
        sft_inputs["input_latents"] = sft_latents
        sft_loss = FlowMatchSFTLoss(self.pipe, **sft_inputs)

        vdpo_inputs = {k: v for k, v in base_inputs.items() if k != "context"}
        vdpo_inputs["context"] = vdpo_context
        vdpo_inputs["input_latents_chosen"] = vdpo_chosen_latents
        vdpo_inputs["input_latents_rejected"] = vdpo_rejected_latents
        vdpo_loss = FlowMatchDPOLoss(
            self.pipe, ref_dit=self._get_dpo_ref_dit(),
            dpo_beta=self.dpo_beta, **vdpo_inputs
        )

        loss = mask_dpo_loss + sft_loss + vdpo_loss
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--preview_steps", type=int, default=0, help="Generate a preview video every N optimizer steps. Set 0 to disable.")
    parser.add_argument("--preview_prompt", type=str, default=None, help="Prompt used for periodic preview generation.")
    parser.add_argument("--preview_negative_prompt", type=str, default="", help="Negative prompt used for periodic preview generation.")
    parser.add_argument("--preview_num_inference_steps", type=int, default=8, help="Inference steps used for each preview video.")
    parser.add_argument("--preview_num_frames", type=int, default=81, help="Number of frames in each preview video.")
    parser.add_argument("--preview_height", type=int, default=None, help="Height for preview video. Defaults to training height, then 480.")
    parser.add_argument("--preview_width", type=int, default=None, help="Width for preview video. Defaults to training width, then 832.")
    parser.add_argument("--preview_seed", type=int, default=0, help="Random seed used for periodic preview generation.")
    parser.add_argument("--preview_cfg_scale", type=float, default=1.0, help="CFG scale used for periodic preview generation.")
    parser.add_argument("--preview_fps", type=float, default=15.0, help="FPS of saved preview videos.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if not hasattr(args, "dpo_ref_model_path"):
        args.dpo_ref_model_path = None
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    video_operator = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4 if not args.framewise_decoding else 1,
        time_division_remainder=1 if not args.framewise_decoding else 0,
    )
    if args.task.startswith("maskdpo"):
        mask_operator = build_mask_operator(
            base_path=args.dataset_base_path,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fallback_video_operator=video_operator,
        )
        dataset = MaskDPOVideoDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            video_operator=video_operator,
            mask_operator=mask_operator,
        )
    elif args.task.startswith("dpo"):
        dataset = DPOVideoDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            video_operator=video_operator,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=video_operator,
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
                "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
            }
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dpo_beta=args.dpo_beta,
        dpo_ref_model_path=args.dpo_ref_model_path,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        preview_steps=args.preview_steps,
        preview_kwargs={
            "prompt": args.preview_prompt,
            "negative_prompt": args.preview_negative_prompt,
            "num_inference_steps": args.preview_num_inference_steps,
            "num_frames": args.preview_num_frames,
            "height": args.preview_height if args.preview_height is not None else args.height,
            "width": args.preview_width if args.preview_width is not None else args.width,
            "seed": args.preview_seed,
            "cfg_scale": args.preview_cfg_scale,
            "fps": args.preview_fps,
        },
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "dpo:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "dpo": launch_dpo_training_task,
        "dpo:train": launch_dpo_training_task,
        "maskdpo": launch_dpo_training_task,
        "maskdpo:train": launch_dpo_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)

