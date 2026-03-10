#!/bin/bash

folder_path="/nfs/scratch1/ajain/spatialdino/scripts/vols_to_save/timepoints" #"/nfs/scratch1/ajain/spatialdino/nucleus2_results/"
number_of_files=-1 # -1 for all timepoints, otherwise chose a number
save_path="/raid1/cme_tests/results/ablations/ants"
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export NUM_PROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

torchrun --nnodes 1 --node_rank 0 --nproc_per_node $NUM_PROC_PER_NODE --rdzv_endpoint=localhost:9999 ./scripts/inference/inference.py \
  file_path="$folder_path" \
  save_path="$save_path" \
  number_of_files=$number_of_files \
  crop_params="[0,0,0,0,0,0]"
