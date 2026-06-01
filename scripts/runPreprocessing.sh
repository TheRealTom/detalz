#!/bin/bash

# ------------------------------------------------------------------------------
# PBS DIRECTIVES
# ------------------------------------------------------------------------------
# Name of the job
#PBS -N EEG_MNE_Preproc
#PBS -l select=1:ncpus=16:mem=48gb:scratch_local=10gb
#PBS -l walltime=01:00:00

# Join standard output and standard error into a single log file
#PBS -j oe

# ------------------------------------------------------------------------------
# SCRIPT EXECUTION
# ------------------------------------------------------------------------------

# 1. Setup trap to clean the scratch directory when the job finishes or fails
trap 'clean_scratch' TERM EXIT

# Check if SCRATCHDIR is allocated
if [ -z "$SCRATCHDIR" ] ; then
    echo "Error: SCRATCHDIR is not set!"
    exit 1
fi

# 2. Load necessary modules
# Load a Python module (check `module avail python` on frontend for latest versions)
module add python/3.10.4-gcc-11.2.0-b5kma2x

# 3. Navigate to the directory where you submitted the job
cd $PBS_O_WORKDIR

# 4. Activate your Python virtual environment where MNE is installed
# (Replace 'mne_env' with the path to your actual virtual environment)
source mne_env/bin/activate

# 5. Copy your Python script and EEG data to the fast local scratch directory
echo "Copying files to scratch..."
cp preprocess_pipeline.py $SCRATCHDIR/
cp -r raw_eeg_data/ $SCRATCHDIR/

# 6. Move to the scratch directory to run the processing
cd $SCRATCHDIR

# 7. Run the Python MNE script
# (Tip: MNE can utilize multiple cores. If your script supports it, 
# you can use the $PBS_NUM_PPN variable to set n_jobs dynamically)
echo "Starting MNE preprocessing..."
python preprocess_pipeline.py 

# 8. Copy the processed results back to your home/storage directory
echo "Copying results back..."
# Ensure the destination folder exists
mkdir -p $PBS_O_WORKDIR/processed_results
cp -r output_data/* $PBS_O_WORKDIR/processed_results/

echo "Job finished successfully."