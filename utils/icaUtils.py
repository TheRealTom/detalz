import mne

from amica_python import fit_ica

def get_amica_transformer(raw: any, n_components: int):
    ica = fit_ica(raw, n_components=n_components, max_iter=10)
    return ica

def get_mne_transformer(**kwargs):
    n_components = mne.preprocessing.PCA()
    ica = mne.preprocessing.ICA(**kwargs)
    return ica
