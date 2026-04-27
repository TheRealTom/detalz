import mne
from mne_icalabel import label_components

import os

def process_file(data_path: str, 
                 l_freq=0.5, 
                 h_freq=40.0,
                 notch_freq=50.0, 
                 n_components=25, 
                 epoch_duration=10.0, 
                 min_confidence=0.70,
                 debug=False):
     # ---------------------------------------------------------
    # 1. Load the BrainVision Data
    # ---------------------------------------------------------
    # MNE only needs the path to the .vhdr file. 
    # It will automatically find the corresponding .vmrk and .eeg files in the same folder.

    if debug:
        print("Loading data...")
    
    raw = mne.io.read_raw_brainvision(vhdr_fname=data_path, preload=True)

    # (Optional but recommended) Set a standard montage so MNE knows where the electrodes are
    # raw.set_montage('standard_1020')

    # ---------------------------------------------------------
    # 2. Filtering
    # ---------------------------------------------------------
    if debug:
        print("Applying filters...")
    # Apply a bandpass filter (combines 0.5 Hz high-pass and 40 Hz low-pass)
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Apply a Notch filter at 50 Hz to remove European power line noise
    raw.notch_filter(freqs=notch_freq)
    
    # ---------------------------------------------------------
    # NEW: Common Average Reference (CAR)
    # ---------------------------------------------------------
    if debug:
        print("Applying Common Average Reference...")
    # 'average' sets the reference to the mean of all EEG channels
    # projection=True computes it as a spatial projection (standard MNE practice)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj() # Apply the projection immediately

    # ---------------------------------------------------------
    # 3. Independent Component Analysis (ICA)
    # ---------------------------------------------------------
    if debug:
        print("Fitting ICA...")
    # Initialize ICA. 15 components is a safe default, adjust based on your channel count.
    ica = mne.preprocessing.ICA(method="picard", fit_params=dict(ortho=False, extended=True), n_components=n_components, random_state=42, max_iter='auto', verbose="debug")
    ica.fit(raw)

    print("Running ICLabel to detect artifacts...")
    # Pass the raw data and the fitted ICA object to the machine learning model
    ic_labels = label_components(raw, ica, method='iclabel')

    # Extract the predicted labels and their confidence scores
    labels = ic_labels['labels']
    probabilities = ic_labels['y_pred_proba']

    # Define the types of artifacts you want the script to remove
    artifacts_to_remove = ['eye blink', 'muscle artifact', 'heart beat']
    bad_components = []

    # Loop through all components and check their label and confidence
    for idx, (label, prob) in enumerate(zip(labels, probabilities)):
        if label in artifacts_to_remove and prob >= min_confidence:
            print(f" -> Automatically flagged Component {idx} as '{label}' (Confidence: {prob*100:.1f}%)")
            bad_components.append(idx)

    # Assign the bad components to the ICA's exclude list
    ica.exclude = bad_components.copy()

    # Apply the ICA spatial weights back to the raw data
    raw_ica = raw.copy()
    ica.apply(raw_ica)

    # ---------------------------------------------------------
    # 4 & 5. Epoching (10 seconds) and Detrending
    # ---------------------------------------------------------
    if debug:
        print("Epoching and Detrending...")
    # Create artificial events every 10 seconds for continuous epoching
    events = mne.make_fixed_length_events(raw_ica, duration=epoch_duration)

    # Create the epochs
    # - tmin=0 and tmax=10 creates exactly 10-second windows.
    # - detrend=1 applies a linear detrend to each epoch individually.
    # - baseline=None is used because detrending usually replaces standard baseline correction.
    epochs = mne.Epochs(
        raw_ica,
        events,
        tmin=0.0,
        tmax=epoch_duration,
        baseline=None,
        detrend=1, 
        preload=True
    )

    if debug:
        print(f"Pipeline complete! Created {len(epochs)} {str(epoch_duration)}-second epochs.")

    # You can now save the processed epochs to a new file
    # epochs.save('processed_data-epo.fif', overwrite=True)

if __name__ == "__main__":
    print("ok")