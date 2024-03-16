#!/bin/bash

python prepare_data.py \
--parent_directory /nfs/datasync4/tklab-llsm \
--search_pattern "**/DS" \
--directories_up 2 \
--save_path /nfs/datasync4/alavaee/tklab-llsm \
--found_directories_limit 10 \
--min_tif_files_in_folder 20 \
--max_workers 100