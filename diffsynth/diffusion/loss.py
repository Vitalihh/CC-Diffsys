from .base_pipeline import BasePipeline
import torch

def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # noise-x_0
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) # [1 C F H W]
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    # 只支持batch_size=1，>1会出错
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss

# 增加dpo loss
def FlowMatchDPOLoss(pipe: BasePipeline, ref_dit: torch.nn.Module = None, dpo_beta=500.0, **inputs):
    if ref_dit is None:
        raise ValueError("`ref_dit` is required for DPO loss.")
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents_chosen = inputs["input_latents_chosen"]
    input_latents_rejected = inputs["input_latents_rejected"]

    noise = torch.randn_like(input_latents_chosen) # [C F H W]
    if input_latents_rejected.shape != input_latents_chosen.shape:
        raise ValueError("chosen video not match with rejected video")
    
    #偏好对用相同的noise
    noisy_chosen = pipe.scheduler.add_noise(input_latents_chosen, noise, timestep)
    target_chosen = pipe.scheduler.training_target(input_latents_chosen, noise, timestep)

    noisy_rejected = pipe.scheduler.add_noise(input_latents_rejected, noise, timestep)
    target_rejected = pipe.scheduler.training_target(input_latents_rejected, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    inputs_base = {k: v for k, v in inputs.items() if k not in (
        "input_latents_chosen", "input_latents_rejected", "input_latents", "dpo_beta",
    )}

    inputs_chosen = {**inputs_base, "latents": noisy_chosen}
    inputs_rejected = {**inputs_base, "latents": noisy_rejected}

    if "first_frame_latents" in inputs:
        inputs_chosen["latents"][:, :, 0:1] = inputs["first_frame_latents"]
        inputs_rejected["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    pred_chosen = pipe.model_fn(**models, **inputs_chosen, timestep=timestep) # [C F H W]
    pred_rejected = pipe.model_fn(**models, **inputs_rejected, timestep=timestep)

    # 这里只替换被训练模型的dit作为ref_model
    models_ref = dict(models)
    models_ref["dit"] = ref_dit

    with torch.no_grad():
        ref_pred_chosen = pipe.model_fn(**models_ref, **inputs_chosen, timestep=timestep)
        ref_pred_rejected = pipe.model_fn(**models_ref, **inputs_rejected, timestep=timestep)

    # 对于T2V可以不管
    if "first_frame_latents" in inputs:
        pred_chosen = pred_chosen[:, :, 1:]
        pred_rejected = pred_rejected[:, :, 1:]
        ref_pred_chosen = ref_pred_chosen[:, :, 1:]
        ref_pred_rejected = ref_pred_rejected[:, :, 1:]
        target_chosen = target_chosen[:, :, 1:]
        target_rejected = target_rejected[:, :, 1:]

    # 只支持batch_size=1，直接mean()不支持batch_size>1
    # 这里根据Flow-DPO没有乘weight
    model_loss_chosen = ((pred_chosen.float() - target_chosen.float())**2).mean()
    model_loss_rejected = ((pred_rejected.float() - target_rejected.float())**2).mean()
    model_diff = model_loss_chosen - model_loss_rejected
    ref_loss_chosen = ((ref_pred_chosen.float() - target_chosen.float())**2).mean()
    ref_loss_rejected = ((ref_pred_rejected.float() - target_rejected.float())**2).mean() 
    ref_diff = ref_loss_chosen - ref_loss_rejected

    scale_term = -0.5 * dpo_beta
    inside_term = scale_term * (model_diff-ref_diff)
    loss = -torch.nn.functional.logsigmoid(inside_term)

    return loss


# 增加mask dpo loss，只在mask区域计算DPO loss
def FlowMatchMaskDPOLoss(pipe: BasePipeline, ref_dit: torch.nn.Module = None, dpo_beta=500.0, mask=None, **inputs):
    if ref_dit is None:
        raise ValueError("`ref_dit` is required for Mask DPO loss.")
    if mask is None:
        raise ValueError("`mask` is required for Mask DPO loss.")

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents_chosen = inputs["input_latents_chosen"]
    input_latents_rejected = inputs["input_latents_rejected"]
    if "strength" not in inputs:
        raise ValueError("`strength` is required for Mask DPO loss.")
    strength = inputs["strength"]
    if strength is None:
        raise ValueError("`strength` is None in Mask DPO loss.")

    noise = torch.randn_like(input_latents_chosen)
    if input_latents_rejected.shape != input_latents_chosen.shape:
        raise ValueError("chosen video not match with rejected video")

    noisy_chosen = pipe.scheduler.add_noise(input_latents_chosen, noise, timestep)
    target_chosen = pipe.scheduler.training_target(input_latents_chosen, noise, timestep)

    noisy_rejected = pipe.scheduler.add_noise(input_latents_rejected, noise, timestep)
    target_rejected = pipe.scheduler.training_target(input_latents_rejected, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    inputs_base = {k: v for k, v in inputs.items() if k not in (
        "input_latents_chosen", "input_latents_rejected", "input_latents", "dpo_beta", "strength",
    )}

    inputs_chosen = {**inputs_base, "latents": noisy_chosen}
    inputs_rejected = {**inputs_base, "latents": noisy_rejected}
    
    pred_chosen = pipe.model_fn(**models, **inputs_chosen, timestep=timestep)
    pred_rejected = pipe.model_fn(**models, **inputs_rejected, timestep=timestep)

    models_ref = dict(models)
    models_ref["dit"] = ref_dit

    with torch.no_grad():
        ref_pred_chosen = pipe.model_fn(**models_ref, **inputs_chosen, timestep=timestep)
        ref_pred_rejected = pipe.model_fn(**models_ref, **inputs_rejected, timestep=timestep)

    # [1 C F H W]
    mask = mask.to(device=pred_chosen.device, dtype=torch.float32)
    if mask.shape != pred_chosen.shape:
        raise ValueError("Mask shape error")
    mask_sum = mask.sum().clamp(min=1.0)

    # 只在mask区域计算MSE
    model_loss_chosen = ((pred_chosen.float() - target_chosen.float())**2 * mask).sum() / mask_sum
    model_loss_rejected = ((pred_rejected.float() - target_rejected.float())**2 * mask).sum() / mask_sum
    model_diff = model_loss_chosen - model_loss_rejected

    ref_loss_chosen = ((ref_pred_chosen.float() - target_chosen.float())**2 * mask).sum() / mask_sum
    ref_loss_rejected = ((ref_pred_rejected.float() - target_rejected.float())**2 * mask).sum() / mask_sum
    ref_diff = ref_loss_chosen - ref_loss_rejected

    alpha = (strength-0.75) / (0.95-0.75)
    scale_term = -0.5 * dpo_beta * (1+alpha)
    inside_term = scale_term * (model_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(inside_term)

    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
