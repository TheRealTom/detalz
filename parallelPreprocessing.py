import time
import logging
import os
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple

# Import your improved function
from preprocessing import process_file

# --- Environment Setup ---
load_dotenv()
DATASET_DIR = Path(os.getenv("DATASET_DIR", "./default_dataset/"))

# We now have two separate log files
CSV_LOG_FILE = Path(os.getenv("CSV_LOG_FILE", "preprocessing_time_log.csv"))
EXECUTION_LOG_FILE = Path(os.getenv("EXECUTION_LOG_FILE", "preprocessing_output.log"))

# Max workers
MAX_WORKERS=int(os.getenv("MAX_WORKERS", 1))

# Data files regex
DATAFILES_REGEX=os.getenv("DATAFILES_REGEX", "./*.vhdr")

# --- Logging Configuration ---
# This configures all logger instances to write to BOTH the console and a text file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(EXECUTION_LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def worker_task(filepath: Path) -> Tuple[str, float, str]:
    """
    Wrapper function for a parallel worker. 
    
    Parameters
    ----------
    filepath : Path
        Path to the .vhdr file.

    Returns
    -------
    Tuple[str, float, str]
        Subject ID, execution time, and status (Success/Error).
    """
    subject_id = filepath.stem  # Filename without extension
    start_time = time.perf_counter()
    
    try:
        # Call the improved function from preprocessing.py
        process_file(str(filepath), debug=False, mne_debug="error")
        status = "Success"
    except Exception as e:
        status = f"Error: {str(e)}"
        logger.error(f"  -> [FAILED] {subject_id}: {e}")
        
    execution_time = time.perf_counter() - start_time
    return subject_id, execution_time, status

def run_parallel_pipeline(search_pattern: str = DATAFILES_REGEX):
    """
    Main control function for parallel processing of the dataset.
    """
    vhdr_files = list(DATASET_DIR.glob(search_pattern))
    
    if not vhdr_files:
        logger.warning(f"No files found in: {DATASET_DIR} matching pattern {search_pattern}")
        return

    logger.info(f"Found {len(vhdr_files)} files. Starting ProcessPoolExecutor...")

    total_start_time = time.perf_counter()
    
    # Open the CSV log file strictly for timing and status tracking
    with open(CSV_LOG_FILE, "w", encoding="utf-8") as csv_log:
        csv_log.write("subject_id,execution_time_seconds,status\n")
        
        # Utilize maximum available CPU cores
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map tasks to workers
            future_to_file = {executor.submit(worker_task, f): f for f in vhdr_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                sub_id, duration, status = future.result()
                
                # Console and text file output via logger
                log_msg = f"Completed: {sub_id} | Time: {duration:.2f}s | Status: {status}"
                if status == "Success":
                    logger.info(f"{log_msg}")
                else:
                    logger.error(f"{log_msg}")
                
                # Continuous CSV writing for time tracking
                csv_log.write(f"{sub_id},{duration:.6f},{status}\n")
                csv_log.flush()

        total_duration = time.perf_counter() - total_start_time
        csv_log.write(f"TOTAL_PIPELINE,{total_duration:.6f},Finished\n")
        
    logger.info(f"Processing complete. Total time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    run_parallel_pipeline()