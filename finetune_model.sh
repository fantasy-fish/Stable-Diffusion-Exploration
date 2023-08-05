export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="zhang_xiaogang-out"
export HUB_MODEL_ID="fantasyfish/zhang_xiaogang-lora"
export DATA_DIR="/home/fantasyfish/Desktop/visual_electric/hacktogether-shawn/data/zhang_xiaogang/train"
export HUB_TOKEN="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"

accelerate launch --mixed_precision="fp16"  examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --hub_token=$HUB_TOKEN \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="a painting of a girl dancing on top of a table" \
  --seed=1337

