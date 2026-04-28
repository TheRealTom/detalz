import os
import time
from pathlib import Path
from dotenv import load_dotenv
from preprocessing import process_file

def run_pipeline() -> None:
    load_dotenv()

    # Konfigurace cest
    dataset_dir: str = os.getenv("DATASET_DIR", "./default_dataset/")
    csv_log_file_path: str = os.getenv("CSV_LOG_FILE", "processing_log.csv")
    
    dataset_path = Path(dataset_dir)
    datafiles_regex =os.getenv("DATAFILES_REGEX", "./*.vhdr")
    vhdr_files = list(dataset_path.glob(datafiles_regex))

    if not vhdr_files:
        print(f"No files found in {dataset_path} with pattern {datafiles_regex}")
        return

    print(f"Found {len(vhdr_files)} files. Starting sequential processing...\n")

    total_start_time = time.perf_counter()

    with open(csv_log_file_path, "w", encoding="utf-8") as log:
        log.write("subject_id,execution_time_seconds,status\n")
        
        for filepath in vhdr_files:
            subject_id = filepath.parent.name
            print(f"Processing: {subject_id}...")
            
            start_time = time.perf_counter()
            
            # Volání preprocessingu
            success = process_file(str(filepath), debug=False)
            status = "Success" if success else "Error"
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            print(f"Finished {subject_id} in {duration:.2f}s [{status}]\n")
            
            # Zápis do logu
            log.write(f"{subject_id},{duration:.6f},{status}\n")
            log.flush() # Okamžitý zápis na disk

        total_duration = time.perf_counter() - total_start_time
        log.write(f"TOTAL_PIPELINE,{total_duration:.6f},Finished\n")

        print(f"--- Pipeline finished ---")
        print(f"Total time: {total_duration:.2f} seconds.")
        

if __name__ == "__main__":
    run_pipeline()