#!/bin/bash

mkdir -p pretrained/t2m/Dec
mkdir -p pretrained/t2m/Trans

# Download the model files using gdown
gdown "https://drive.google.com/uc?id=15K-kOJDTtF_jzncQGC84FZw1rUxuRUKk" -O pretrained/t2m/Dec/model.pth
gdown "https://drive.google.com/uc?id=1YwivaohyNZSWaCTNX1REPxvA-bpLzz_K" -O pretrained/t2m/Trans/model.pth 

echo "Download complete."