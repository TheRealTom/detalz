import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, Optional, Union
from scipy.stats import kurtosis, skew


def extract_spatial_metadata(data: Union[mne.io.BaseRaw, mne.Epochs], prefix: str = "") -> Dict[str, Any]:
    """
    Extracts montage and electrode localization information.
    
    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data object.
    prefix : str
        Prefix for dictionary keys.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with spatial and montage characteristics.
    """
    info = data.info
    chs = info['chs']
    
    # Check if 3D coordinates are available
    has_loc = any([not np.all(ch['loc'][:3] == 0) for ch in chs if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])
    
    avg_dist = 0.0
    if has_loc:
        locs = np.array([ch['loc'][:3] for ch in chs if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])
        from scipy.spatial.distance import pdist
        avg_dist = float(np.mean(pdist(locs)))

    return {
        f"{prefix}montage_set": 1 if has_loc else 0,
        f"{prefix}n_eeg_channels": info['nchan'],
        f"{prefix}avg_electrode_dist": avg_dist,
        f"{prefix}coord_frame": info.get('dev_head_t', {}).get('to', 'unknown') if info.get('dev_head_t') else 'unknown'
    }

def extract_hardware_metadata(data: Union[mne.io.BaseRaw, mne.Epochs], prefix: str = "") -> Dict[str, Any]:
    """
    Extracts recording parameters, filters, and hardware status.
    """
    info = data.info
    return {
        f"{prefix}sfreq": info['sfreq'],
        f"{prefix}highpass": info['highpass'],
        f"{prefix}lowpass": info['lowpass'],
        f"{prefix}line_freq": info.get('line_freq', "N/A"),
        f"{prefix}n_bad_channels": len(info['bads'])
    }

def calculate_advanced_quality_metrics(data_obj: Union[mne.io.BaseRaw, mne.Epochs], prefix: str = "") -> Dict[str, float]:
    """
    Calculates higher-order statistical moments to quantify noise and artifacts.
    """
    # Get pure numpy data (average across epochs if epoched)
    if isinstance(data_obj, mne.Epochs):
        data = data_obj.get_data().mean(axis=0) # Shape: (channels, times)
    else:
        data = data_obj.get_data(picks='eeg')

    return {
        f"{prefix}var_mean": float(np.var(data, axis=1).mean()),
        f"{prefix}kurtosis_mean": float(kurtosis(data, axis=1).mean()),
        f"{prefix}skewness_mean": float(skew(data, axis=1).mean()),
        f"{prefix}max_ptp": float(np.ptp(data, axis=1).max()),
    }

def get_spectral_metrics(data: Union[mne.io.BaseRaw, mne.Epochs], 
                         prefix: str = "",
                         bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Calculates Absolute and Relative Spectral Power.
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

    n_fft = min(2048, len(data.times))
    spectrum = data.compute_psd(method='welch', fmin=1.0, fmax=45.0, n_fft=n_fft, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    axis_to_mean = (0, 1) if isinstance(data, mne.Epochs) else 0
    avg_psd = psds.mean(axis=axis_to_mean)
    total_power = np.sum(avg_psd)
    
    metrics = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_pow = np.sum(avg_psd[idx])
        metrics[f"{prefix}{band}_abs"] = float(band_pow)
        metrics[f"{prefix}{band}_rel"] = float(band_pow / total_power) if total_power > 0 else 0.0
        
    return metrics

def get_comprehensive_comparison(raw: mne.io.BaseRaw, epochs: mne.Epochs, subject_id: str) -> Dict[str, Any]:
    """
    Assembles the final dictionary with all available metrics for both Raw and Preprocessed data.
    """
    final_stats = {"subject_id": subject_id}
    
    # Iterate through both data states to extract features
    for label, obj in [("raw_", raw)]:
        final_stats.update(extract_hardware_metadata(obj, label))
        final_stats.update(extract_spatial_metadata(obj, label))
        final_stats.update(calculate_advanced_quality_metrics(obj, label))
        final_stats.update(get_spectral_metrics(obj, label))
        
    return final_stats

if __name__ == "__main__":
    print("Initializing Statistical Analysis Pipeline...")
    
    # 1. Load configuration from .env
    load_dotenv()
    DATASET_DIR = os.getenv("DATASET_DIR", "./default_dataset/")
    OUTPUT_CSV = os.getenv("CSV_STATS", "subject_statistics.csv")
    
    dataset_path = Path(DATASET_DIR)
    # Using the same search pattern as parallelPreprocessing.py
    search_pattern = "sub-EXCI*/sub-EXCI*_ses-1_task-rest_eeg.vhdr"
    vhdr_files = list(dataset_path.glob(search_pattern))
    
    if not vhdr_files:
        print(f"[WARNING] No .vhdr files found in {dataset_path} matching {search_pattern}")
    else:
        print(f"Found {len(vhdr_files)} files for analysis.\n")
        
        all_subjects_stats = []
        
        # 2. Iterate and Process
        for filepath in vhdr_files:
            subject_id = filepath.parent.name
            print(f"Analyzing {subject_id}...")
            
            try:
                raw = mne.io.read_raw_brainvision(vhdr_fname=filepath, verbose="error")
                # Extract all statistics
                stats = get_comprehensive_comparison(raw, [], subject_id)
                all_subjects_stats.append(stats)
                del raw
                print(f"  -> [SUCCESS] Stats extracted.")
                
            except Exception as e:
                print(f"  -> [FAILED] Error processing {subject_id}: {e}")
                
        # 3. Export to CSV
        if all_subjects_stats:
            df = pd.DataFrame(all_subjects_stats)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n--- Analysis Complete ---")
            print(f"Successfully processed {len(df)} subjects.")
            print(f"Results saved to: {OUTPUT_CSV}")