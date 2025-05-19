#!/bin/bash

# Merge LoRA weights into the base model

# Hardcoded paths
MODEL_PATH="/mnt/dunghd/vllm/LLaVA-Med/checkpoints_final_qa_add/llm-med-mistral-v1.5-7b/checkpoint-300"  # Path to the LoRA model
MODEL_BASE="llava-med-v1.5-mistral-7b"  # Path to the base model
SAVE_MODEL_PATH="checkpoints_final_qa_add/llava-med-mistral-v1.5-7b"  # Path to save the merged model

# Run the Python script to merge the weights
python -m llava.scripts.merge_lora_weights \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --save-model-path $SAVE_MODEL_PATH

echo "LoRA weights merged successfully! The merged model is saved at $SAVE_MODEL_PATH"
