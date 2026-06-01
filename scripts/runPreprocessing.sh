#!/bin/bash
#PBS -N eeg_amica_gpu_100GB
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:scratch_local=300gb
#PBS -l walltime=24:00:00
#PBS -j oe

# Nastavení prostředí
module add python/3.11.11
module add cuda/12.6.2

# Definice síťových cest (trvalé úložiště)
NETWORK_DATA_DIR="/storage/brno2/home/$USER/eeg_project/data"
NETWORK_OUTPUT_DIR="/storage/brno2/home/$USER/eeg_project/output"

# Kontrola, zda nám byl přidělen SCRATCHDIR
if [ -z "$SCRATCHDIR" ] || [ ! -d "$SCRATCHDIR" ]; then
    echo "Kritická chyba: SCRATCHDIR nebyl alokován." >&2
    exit 1
fi

# ---------------------------------------------------------
# KROK A: Příprava a kopírování dat na lokální NVMe výpočetního uzlu
# ---------------------------------------------------------
echo "Kopíruji 100GB dataset do lokálního scratch: $SCRATCHDIR"
# Kopírujeme data a zachováváme strukturu složek
cp -r $NETWORK_DATA_DIR/* $SCRATCHDIR/

# Vytvoření lokální výstupní složky
LOCAL_OUTPUT_DIR="$SCRATCHDIR/output"
mkdir -p $LOCAL_OUTPUT_DIR

# ---------------------------------------------------------
# KROK B: Aktivace Python prostředí a běh výpočtu
# ---------------------------------------------------------
cd $PBS_O_WORKDIR
python -m venv venv_gpu
source venv_gpu/bin/activate
pip install --upgrade pip
pip install mne mne-icalabel pydantic python-dotenv cupy-cuda12x

# Nastavení proměnných prostředí pro Python skript tak, 
# aby pracoval POUZE s lokálními cestami na uzlu
export DATASET_DIR=$SCRATCHDIR
export OUTPUT_DIR=$LOCAL_OUTPUT_DIR
export LOG_FILE=$LOCAL_OUTPUT_DIR/processing_log.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Spouštím paralelní preprocessing na GPU..."
python parallelPreprocessing.py

# ---------------------------------------------------------
# KROK C: Úklid a přesun výsledků zpět na síťový disk
# ---------------------------------------------------------
echo "Zpracování dokončeno, kopíruji výsledky zpět na domovský svazek..."
# Přesuneme vytvořená data a logy zpět, pokud se skript nezhroutil
cp -r $LOCAL_OUTPUT_DIR/* $NETWORK_OUTPUT_DIR/

# Smazání scratch adresáře z důvodu šetření místa pro další uživatele (dobrý mrav)
rm -rf $SCRATCHDIR/*
echo "Job kompletně hotov."