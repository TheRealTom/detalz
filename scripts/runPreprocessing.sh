#!/bin/bash
#PBS -N eeg_amica_gpu_100GB
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:scratch_local=300gb
#PBS -l walltime=24:00:00
#PBS -j oe

# Nastavení prostředí
module load python/3.11.11

cp -r ${PBS_O_WORKDIR} ${SCRATCHDIR}

cd ..
mkdir ${SCRATCHDIR}/data
mkdir ${SCRATCHDIR}/output

cp -r ./data/sub-EXCI0103/ ${SCRATCHDIR}/data/

cd ${SCRATCHDIR}
cd repo
python -m venv venv_eeg
source ./venv_eeg/bin/activate

export TMPDIR=$SCRATCHDIR

pip install .
# Vytvoření lokální výstupní složky
LOCAL_OUTPUT_DIR="$SCRATCHDIR/output"
mkdir -p $LOCAL_OUTPUT_DIR

# ---------------------------------------------------------
# KROK B: Aktivace Python prostředí a běh výpočtu
# ---------------------------------------------------------
# Nastavení proměnných prostředí pro Python skript tak, 
# aby pracoval POUZE s lokálními cestami na uzlu
export DATASET_DIR=$SCRATCHDIR
export OUTPUT_DIR=$LOCAL_OUTPUT_DIR
export LOG_FILE=$LOCAL_OUTPUT_DIR/processing_log.txt

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

echo "Spouštím paralelní preprocessing na GPU..."
python parallelPreprocessing.py

# ---------------------------------------------------------
# KROK C: Úklid a přesun výsledků zpět na síťový disk
# ---------------------------------------------------------
echo "Zpracování dokončeno, kopíruji výsledky zpět na domovský svazek..."
# Přesuneme vytvořená data a logy zpět, pokud se skript nezhroutil
mkdir ${PBS_O_WORKDIR}/output
cp -r $LOCAL_OUTPUT_DIR/* ${PBS_O_WORKDIR}/output

# Smazání scratch adresáře z důvodu šetření místa pro další uživatele (dobrý mrav)
clean_scratch
rm -rf $SCRATCHDIR/*
echo "Job kompletně hotov."