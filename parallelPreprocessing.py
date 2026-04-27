from preprocessing import process_file
import time

DATASET_1="../../PhD/datasets/"

if __name__ == "__main__":
    # Start the timer
    start_time = time.perf_counter()
    
    # TASK
    process_file(DATASET_1 + "sub-EXCI0103/sub-EXCI0103_ses-1_task-rest_eeg.vhdr", debug=True, n_components=25)
    
    # Stop the timer
    end_time = time.perf_counter()

    # Calculate the Delta time
    execution_time = end_time - start_time
    print(f"Function took {execution_time:.6f} seconds to run.")