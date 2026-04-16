accelerate launch --config_file config.yaml train.py \
  --task "dpo" \
  --model_paths '[
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
]' \
  --tokenizer_path "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B-Diffusers/tokenizer" \
  --dataset_base_path "/home/gaofanding/maskdpo-Wan/sample/test_2" \
  --dataset_metadata_path "/home/gaofanding/maskdpo-Wan/sample/test_dpo_0412.json" \
  --output_path "/home/gaofanding/maskdpo-Wan/dpo_lora/0413" \
  --lora_base_model "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --dataset_repeat 1 \
  --dataset_num_workers 2 \
  --learning_rate 1e-5 \
  --dpo_beta 500.0 \
  --num_epochs 10 \
  --save_steps 10 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --preview_steps 20 \
  --preview_prompt "a cat sitting on a boat" \
  --preview_num_inference_steps 50 \
  --preview_num_frames 81 \
  --preview_fps 16 \
  # 续训设置--resume_step 40 --max_train_steps 45
  # 数据格式在/examples/wanvideo/model_training/test_dpo.json