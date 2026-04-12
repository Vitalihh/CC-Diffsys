import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.unified_dataset import DPOVideoDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
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
    ):
        super().__init__()
        # Warning
        # 如果显存够用可以尝试不开
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "dpo:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "dpo": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchDPOLoss(pipe, dpo_beta=self.dpo_beta, **inputs_shared, **inputs_posi),
            "dpo:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchDPOLoss(pipe, dpo_beta=self.dpo_beta, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.dpo_beta = dpo_beta
        self._preview_prompt_warning_emitted = False

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
    
    # 增加dpo inputs 
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
        # 之后可以在parse_extra_inputs增加处理mask
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if self.task.startswith("dpo") and not self.task.endswith(":data_process"):
            return self._forward_dpo(inputs)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def _forward_dpo(self, inputs):
        inputs_shared, inputs_posi, inputs_nega = inputs
        input_video_rejected = inputs_shared.pop("input_video_rejected", None)

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
    if args.task.startswith("dpo"):
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
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
