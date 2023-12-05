#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python tools/train.py custom_configs/final/50_R50.py
CUDA_VISIBLE_DEVICES=1 python tools/train.py custom_configs/final/70_R50.py
CUDA_VISIBLE_DEVICES=1 python tools/train.py custom_configs/final/90_R50.py
