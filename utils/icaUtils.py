import mne

from pyamica import AmicaICA

def get_amica_transformer(device="cpu"):
    ica = AmicaICA(n_components=None, max_iter=2000,
                    device=device, n_models=1)
    return ica

def get_mne_transformer(**kwargs):
    ica = mne.preprocessing.ICA(**kwargs)
    return ica
