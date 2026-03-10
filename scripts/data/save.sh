#!/bin/bash

EXPERIMENT_FILTER_PATTERN="(?i)(dextran|lamp1|npc1|eea1|transferrin|virus|apili)"
DATA_PATHS_TXT="/nfs/scratch2/shared_image_recog_ml/annotations/apili_640.txt"
SAVE_PATH="/raid2/shared_image_recog_ml2/llsm_3d_ds_apili_642"

python save.py \
	--save_3d \
	--use_deskewed_data \
	--auto_crop \
	--min_bbox_ratio 0.6 \
	--max_bbox_ratio 0.8 \
	--dz 0.25 \
	--dy 0.104 \
	--dx 0.104 \
	--z_flatten_mode "NONE" \
	--z_chunk_size -1 \
	--y_chunk_size -1 \
	--x_chunk_size -1 \
	--data_paths_txt $DATA_PATHS_TXT \
	--save_path $SAVE_PATH \
	--experiment_filter_pattern $EXPERIMENT_FILTER_PATTERN \
	--found_experiments_limit 1000 \
	--min_tif_files 20 \
	--max_workers 20