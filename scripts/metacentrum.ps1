#!/bin/bash
# Local execution script to automate the Metacentrum workflow

# --- CONFIGURATION ---
META_USER="xszabo16"
META_SERVER="skirit.metacentrum.cz" 
LOCAL_DATA_DIR="./raw_dataset/"
LOCAL_OUTPUT_DIR="./processed_dataset/"
META_WORK_DIR="/storage/brno2/home/$META_USER/eeg_project"
GITHUB_REPO="https://github.com/TheRealTom/datalz.git"


echo "[1/5] Syncing Raw Data to Metacentrum ($META_SERVER)..."
ssh $META_USER@$META_SERVER "mkdir -p $META_WORK_DIR/data $META_WORK_DIR/output"
rsync -avz --progress $LOCAL_DATA_DIR $META_USER@$META_SERVER:$META_WORK_DIR/data/

echo "[2/5] Setting up Git Repository on Metacentrum..."
ssh $META_USER@$META_SERVER "
    cd $META_WORK_DIR;
    if [ ! -d 'repo' ]; then
        git clone $GITHUB_REPO repo;
    else
        cd repo && git pull;
    fi
"

echo "[3/5] Submitting PBS Job (GPU Queue)..."
JOB_ID=$(ssh $META_USER@$META_SERVER "cat << 'EOF' > $META_WORK_DIR/job_gpu.sh
#!/bin/bash
#PBS -N eeg_amica_gpu
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=02:00:00
#PBS -j oe

# Load necessary modules for Python and GPU
module add python/3.10.4
module add cuda/12.1

# Setup virtual environment
cd $META_WORK_DIR/repo
python -m venv venv_gpu
source venv_gpu/bin/activate

# Install MNE, CuPy (matching the loaded CUDA module) for GPU acceleration
pip install --upgrade pip
pip install mne mne-icalabel pydantic python-dotenv cupy-cuda12x

# NOTE: Make sure your amica_python wrapper is installed here as well
# pip install amica_python (or however you distribute it)

export DATASET_DIR=$META_WORK_DIR/data
export OUTPUT_DIR=$META_WORK_DIR/output
export LOG_FILE=$META_WORK_DIR/output/processing_log.txt

# Force MNE/NumPy to limit thread usage so they don't oversubscribe CPU while waiting for GPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python parallelPreprocessing.py
EOF
qsub $META_WORK_DIR/job_gpu.sh")

echo "Job submitted with ID: $JOB_ID"

echo "[4/5] Waiting for GPU job completion..."
while ssh $META_USER@$META_SERVER "qstat -f $JOB_ID" | grep -q "job_state = [R|Q]"; do
    sleep 60
    echo -n "."
done
echo " Job finished!"

echo "[5/5] Downloading processed data back to Local PC..."
mkdir -p $LOCAL_OUTPUT_DIR
rsync -avz --progress $META_USER@$META_SERVER:$META_WORK_DIR/output/ $LOCAL_OUTPUT_DIR

echo "Pipeline complete. Processed data is in $LOCAL_OUTPUT_DIR."