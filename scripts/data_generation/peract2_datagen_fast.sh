# Paths relative to project root
DATA_PATH=peract2_raw
ZARR_PATH=zarr_datasets/peract2

# Create directories
mkdir -p ${DATA_PATH}
mkdir -p ${ZARR_PATH}

# 1. Download and unzip training data (zarr)
echo "Downloading training data..."
wget -O peract2_zarr.zip https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2.zip
unzip -o peract2_zarr.zip -d ${ZARR_PATH}
rm peract2_zarr.zip

# 2. Download and unzip test seeds (raw)
echo "Downloading test seeds..."
wget -O peract2_test.zip https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
unzip -o peract2_test.zip -d ${DATA_PATH}
rm peract2_test.zip

echo "Data download and extraction complete."
# Good to go!
