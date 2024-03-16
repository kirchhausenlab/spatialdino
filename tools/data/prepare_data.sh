#!/bin/bash

python prepare_data.py /nfs/datasync4/tklab-llsm \
--target_directory DS \
--save_path /nfs/datasync4/alavaee/tklab-llsm \
--found_directories_limit 1 \
--min_tif_files_in_folder 20 \
--max_workers 100