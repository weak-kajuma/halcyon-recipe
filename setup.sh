#! /bin/bash
pip install transformers hf_transfer
pip install sentencepiece datasets evaluate accelerate scikit-learn \
liger_kernel schedulefree wheel wandb bitsandbytes deepspeed
# pip install   tf-keras "numpy<2" 
pip install flash-attn --no-build-isolation
# pip install transformer_engine[pytorch]

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-3.2-1B
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download kajuma/training_02_28 --repo-type dataset