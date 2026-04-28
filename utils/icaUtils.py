import mne

from amica_python import fit_ica

def get_amica_transformer(raw: any):
    ica = fit_ica(raw, max_iter=10)
    return ica

def get_mne_transformer(**kwargs):
    ica = mne.preprocessing.ICA(**kwargs)
    return ica
