accelerate launch --config_file /home/gaofanding/cc_DiffSynth-Studio/examples/wanvideo/model_training/config.yaml train.py \
  --task "sft" \
  --model_paths '[
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
]' \
  --tokenizer_path "/data1/users/gaofanding/ckpts/Wan2.1-T2V-1.3B-Diffusers/tokenizer" \
  --dataset_base_path /home/gaofanding/maskdpo-Wan/sample/test_2 \
  --dataset_metadata_path /home/gaofanding/maskdpo-Wan/sample/test_sft_0412.json \
  --output_path "/home/gaofanding/maskdpo-Wan/sft_lora/0412" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --dataset_num_workers 2 \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --save_steps 10 \
  --gradient_accumulation_steps 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --preview_steps 20 \
  --preview_prompt "a cat sitting on a boat" \
  --preview_num_inference_steps 50 \
  --preview_num_frames 81 \
  --preview_fps 16 \
  # 续训--resume_step 20 --max_train_steps 40
  # 数据集格式在/examples/wanvideo/model_training/test_sft.json