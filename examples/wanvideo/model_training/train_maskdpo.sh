accelerate launch examples/wanvideo/model_training/train.py \
  --task "maskdpo" \
  --dataset_base_path "./data/maskdpo_dataset" \
  --dataset_metadata_path "./data/maskdpo_dataset/metadata.json" \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors" \
  --lora_base_model "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --dataset_repeat 1 \
  --dataset_num_workers 1 \
  --learning_rate 1e-5 \
  --dpo_beta 500.0 \
  --num_epochs 3 \
  --save_steps 20 \
  --output_path "./output/maskdpo_lora" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --preview_steps 20 \
  --preview_prompt "a cat sitting on a boat" \
  --preview_num_inference_steps 50 \
  --preview_num_frames 81 \
  --preview_fps 16

# metadata.json 格式示例:
# [
#   {
#     "prompt": "a cat sitting on a boat",
#     "video_chosen": "chosen/video1.mp4",
#     "video_rejected": "rejected/video1.mp4",
#     "mask": "masks/video1.mp4",
#     "video_sft": "sft/video1.mp4",
#     "video_vdpo_chosen": "vdpo_chosen/video1.mp4",
#     "video_vdpo_rejected": "vdpo_rejected/video1.mp4"
#   }
# ]
# mask为二值掩码视频，白色区域(>0.5)为需要计算mask DPO loss的区域
# 最终loss = mask_dpo_loss + sft_loss + vdpo_loss
