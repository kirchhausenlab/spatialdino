#!/bin/bash

python prepare_data.py /nfs/datasync4/tklab-llsm \
--target_directory DS \
--save_path /nfs/datasync4/alavaee/tklab-llsm \
--max_parent_directories 1 \
--min_tif_files 50 \
--max_search_depth 6 \
--max_workers 1