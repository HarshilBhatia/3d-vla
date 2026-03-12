## STeos

1. get zed docker / apptainer (to avoid any libso issues)
2. don't install via pip! instead go into 
/usr/local/zed.
3. python3 get_python_api.py 

4. you need calibration files for zed-style cameras. (they are not in the droid dataset) 


commands

apptainer shell --nv \
  --bind /work/nvme/bgkz:/work/nvme/bgkz \
  --bind /u/hbhatia1/.local/share/stereolabs/settings:/usr/local/zed/settings \
  zed_4.0.sif

python3 data_processing/extract_svo_depth.py \
    --raw-dir /work/nvme/bgkz/droid_rail_raw \
    --output-dir /work/nvme/bgkz/droid_rail_depths