from preprocessing import process_file
from pathlib import Path

import time

DATASET_1="../../PhD/datasets/"
LOG_FILE = "preprocessing_times.txt"

if __name__ == "__main__":
    dataset_path = Path(DATASET_1)
    search_pattern = "sub-EXCI*/sub-EXCI*_ses-1_task-rest_eeg.vhdr"
    vhdr_files = list(dataset_path.glob(search_pattern))
    print(f"Found {len(vhdr_files)} files. Starting processing...\n")

    with open(LOG_FILE, "w") as log:
        #headers
        log.write("subject_id,execution_time_seconds\n")
        total_start_time = time.perf_counter()
        for filepath in vhdr_files:
            # Extract just the subject folder name (e.g., 'sub-EXCI0104')
            subject_id = filepath.parent.name
            print(f"Processing {subject_id}...")

            # Start the timer
            start_time = time.perf_counter()
            # TASK
            process_file(str(filepath))
            
            # Stop the timer
            end_time = time.perf_counter()

            # Calculate the Delta time
            execution_time = end_time - start_time
            print(f"Function took {execution_time:.6f} seconds to run.")
            log.write(f"{subject_id},{execution_time:.6f}\n")
            log.flush()
        total_end_time = time.perf_counter()
        total_execution = total_end_time - total_start_time
        log.write(f"TOTAL,{execution_time:.6f}\n")
    