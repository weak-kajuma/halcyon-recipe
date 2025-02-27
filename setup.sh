#! /bin/bash
pip install transformers sentencepiece datasets evaluate accelerate scikit-learn \
bitsandbytes deepspeed liger_kernel schedulefree wheel tf-keras "numpy<2" hf_transfer wandb
pip install flash-attn --no-build-isolation
pip install transformer_engine[pytorch]

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-3.2-1B