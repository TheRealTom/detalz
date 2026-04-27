import os
from typing import Optional, Union

import mne
from mne_icalabel import label_components

import utils.icaUtils as iu


def process_file(
    data_path: str,
    output_dir: Optional[str] = None,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    n_components: Union[float, int] = 0.95,
    epoch_duration: float = 10.0,
    min_confidence: float = 0.70,
    overwrite: bool = False,
    debug: bool = False,
    mne_debug: str = "warning"
) -> None:
    """
    Complex pipeline for EEG data preprocessing from BrainVision format.
    Applies filtration, CAR, ICA artefact removal with IC Label detection,
    epoching with linear detrending and saves everything to disk.

    Parameters
    ----------
    data_path : str
        Absolute or relative path to the target `.vhdr` file.
    output_dir : Optional[str], default=None
        Target directory for saving the `.fif` file. If None, 
        the file is saved in the same directory as the source `.vhdr` file.
    l_freq : float, default=0.5
        Lower pass-band edge (High-pass filter) in Hz.
    h_freq : float, default=40.0
        Upper pass-band edge (Low-pass filter) in Hz.
    notch_freq : float, default=50.0
        Frequency to remove power line noise (Notch filter) in Hz.
    n_components : Union[float, int], default=0.95
        Number of components for ICA (int) or percentage of explained variance for PCA (float).
    epoch_duration : float, default=10.0
        Duration of the created epochs in seconds.
    min_confidence : float, default=0.70
        Minimum probability threshold for ICLabel artifact classification.
    overwrite : bool, default=False
        If True, overwrites an existing `-epo.fif` file. If False and the file exists,
        the computation is skipped (Checkpointing).
    debug : bool, default=False
        Enables additional progress printouts to the console.
    mne_debug : str, default="warning"
        Logging level for internal MNE functions (e.g., 'info', 'warning', 'error').

    Raises
    ------
    FileNotFoundError
        If the source `.vhdr` file does not exist.
    RuntimeError
        If a failure occurs during ICA fitting or data processing.
    """

    # Check file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")

    # Make new file for preprocessed epochs
    base_name = os.path.basename(data_path).replace(".vhdr", "-epo.fif")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, base_name)
    else:
        # Defaultní chování: uložení vedle původních dat
        save_path = data_path.replace(".vhdr", "-epo.fif")

    # Checkpoint: If it was done already, we go next
    if os.path.exists(save_path) and not overwrite:
        if debug:
            print(f"[SKIP] Preprocessed data already exist: {save_path}")
        return

    # MNE optimization for CUDA
    try:
        mne.set_config('MNE_USE_CUDA', 'true')
        mne.cuda.init_cuda()
    except (RuntimeError, ImportError):
        pass  # Fallback to CPU

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    if debug:
        print(f"[LOAD] Načítám: {data_path}")
    
    raw = mne.io.read_raw_brainvision(vhdr_fname=data_path, preload=True, verbose=mne_debug)

    # ---------------------------------------------------------
    # 2. Filtration and CAR
    # ---------------------------------------------------------
    if debug:
        print("[FILTER] Aplikuji Bandpass a Notch filtry...")
    
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=mne_debug)
    raw.notch_filter(freqs=notch_freq, verbose=mne_debug)
    
    if debug:
        print("[CAR] Aplikuji Common Average Reference...")
    
    raw.set_eeg_reference('average', projection=True, verbose=mne_debug)
    raw.apply_proj(verbose=mne_debug)

    # ---------------------------------------------------------
    # 3. ICA and artefact detection (ICLabel)
    # ---------------------------------------------------------
    if debug:
        print("[ICA] Fituji ICA model...")
    
    transformer = iu.get_mne_transformer(
        method="picard", 
        fit_params=dict(ortho=False, extended=True),
        n_components=n_components, 
        random_state=42, 
        max_iter='auto'
    )
    transformer.fit(raw, verbose=mne_debug)

    if debug:
        print("[ICLabel] Analyzuji komponenty...")
        
    ic_labels = label_components(raw, transformer, method='iclabel')
    labels = ic_labels['labels']
    probabilities = ic_labels['y_pred_proba']

    artifacts_to_remove = ['eye blink', 'muscle artifact', 'heart beat']
    bad_components = []

    for idx, (label, prob) in enumerate(zip(labels, probabilities)):
        if label in artifacts_to_remove and prob >= min_confidence:
            if debug:
                print(f"  -> Flagged Component {idx}: '{label}' ({prob*100:.1f}%)")
            bad_components.append(idx)

    transformer.exclude = bad_components
    transformer.apply(raw, verbose=mne_debug)

    # ---------------------------------------------------------
    # 4. Epoching and Detrending
    # ---------------------------------------------------------
    if debug:
        print(f"[EPOCHS] Krájím signál po {epoch_duration}s...")
        
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_duration,
        baseline=None,
        detrend=1, 
        preload=True,
        verbose=mne_debug
    )

    # ---------------------------------------------------------
    # 5. Save to disk
    # ---------------------------------------------------------
    if debug:
        print(f"[SAVE] Zapisuji na disk: {save_path}")
        
    # Save to disk
    epochs.save(save_path, overwrite=overwrite, verbose=mne_debug)