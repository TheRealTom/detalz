from preprocessing import process_file
from pathlib import Path

import time
import concurrent.futures

DATASET_1="../../PhD/datasets/"
LOG_FILE = "preprocessing_times.txt"


def process_and_time(filepath: Path):
    subject_id = filepath.parent.name
    
    start_time = time.perf_counter()
    
    try:
        process_file(str(filepath))
        status = "Success"
    except Exception as e:
        status = f"Error"
        print(f"  -> [FAILED] {subject_id}: {e}")
        
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Return the results back to the main thread
    return subject_id, execution_time, status

if __name__ == "__main__":
    dataset_path = Path(DATASET_1)
    search_pattern = "sub-EXCI*/sub-EXCI*_ses-1_task-rest_eeg.vhdr"
    vhdr_files = list(dataset_path.glob(search_pattern))
    print(f"Found {len(vhdr_files)} files. Starting processing...\n")

    with open(LOG_FILE, "w") as log:
        #headers
        log.write("subject_id,execution_time_seconds\n")
        
        #total time
        total_start_time = time.perf_counter()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all files to the worker pool
                # This creates a dictionary mapping each 'future' to its filepath
                futures = {executor.submit(process_and_time, filepath): filepath for filepath in vhdr_files}
                
                # as_completed yields the results as soon as any process finishes
                for future in concurrent.futures.as_completed(futures):
                    # Unpack the returned values from our worker function
                    subject_id, execution_time, status = future.result()
                    
                    print(f"Finished {subject_id} in {execution_time:.2f} seconds. [{status}]")
                    
                    # Write safely to the file from the MAIN process
                    log.write(f"{subject_id},{execution_time:.6f},{status}\n")
                    log.flush()

        total_end_time = time.perf_counter()
        total_execution = total_end_time - total_start_time
        log.write(f"TOTAL,{execution_time:.6f}\n")
    