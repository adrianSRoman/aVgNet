import os
from tqdm import tqdm
import pickle
from random import randrange
import gc
from glob import glob
import h5py
import numpy as np
import scipy
from scipy.io import wavfile

import scipy.constants as constants
import scipy.io.wavfile as wavfile
import scipy.signal.windows as windows
import scipy.linalg as linalg
import skimage.util as skutil
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.spatial as spatial

'''

'''

def extract_visibilities(_data, _rate, T, fc, bw, alpha):
    """
    Transform time-series to visibility matrices.

    Parameters
    ----------
    T : float
        Integration time [s].
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    alpha : float
        Shape parameter of the Tukey window, representing the fraction of
        the window inside the cosine tapered region. If zero, the Tukey
        window is equivalent to a rectangular window. If one, the Tukey
        window is equivalent to a Hann window.

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices (complex-valued).
    """
    N_stft_sample = int(_rate * T)
    if N_stft_sample == 0:
        raise ValueError('Not enough samples per time frame.')
    # print(f'Samples per STFT: {N_stft_sample}')

    N_sample = (_data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = _data.shape[1]
    stf_data = (skutil.view_as_blocks(_data[:N_sample], (N_stft_sample, N_channel))
                .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (N_stf, N_stft_sample, N_channel)
    N_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (N_stf, N_stft_sample, N_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * N_stft_sample / _rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / _rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    # collapsed_spectrum = collapsed_spectrum[0,:]
    S = (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
        collapsed_spectrum.reshape(N_stf, 1, -1))
    return S

def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    '''
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)
        
    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.
        
        # N_slot == number of audio frames in track

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    '''
    S_sti = (extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti)
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
        .squeeze(axis=(1, 2))
        .sum(axis=1))
    return S

def get_visibility_matrix(audio_in, fs):
    # audio_in, fs = self._load_audio(audio_filename)
    
    freq, bw = (skutil  # Center frequencies to form images
        .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
        .mean(axis=-1)), 50.0  # [Hz]

    visibilities = []
    for i in range(9):
        T_sti = 10.0e-3
        T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10
        S = form_visibility(audio_in, fs, freq[i], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]
        visibilities_per_frame = []
        for s_idx in range(N_sample):
            S_D, S_V = linalg.eigh(S[s_idx])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T
            visibilities_per_frame.append(S_norm) 

        visibilities.append(visibilities_per_frame)

    return np.array(visibilities)

def create_full_hdf_data(type='train'):
    eigen_data_path = "/scratch/data/LOCATA/LOCATA_eval_eigenmike_16k/"
    save_mic_data_path = "/scratch/data/LOCATA/MIC_eval_eigenmike_16k/"
    save_path = "/scratch/data/LOCATA/MIC_eval_eigenmike_16k/"

    eigenmike_files = glob("/scratch/data/LOCATA/LOCATA_eval_eigenmike_16k/*.wav")
    print(eigenmike_files)

    with h5py.File(f'{save_path}{type}.hdf',"w") as f:
        mic_data = []
        vg_labels = []
        for clip_name in tqdm(eigenmike_files):
            print("Clipname ", os.path.basename(clip_name))
            fs, eigen_sig = wavfile.read(clip_name)
            print("Sampling rate is", fs)
            vsg_sig = get_visibility_matrix(eigen_sig, fs) # visibility graph matrix 32ch 
            print("Visibility matrix size:", vsg_sig.shape)
            mic_sig = eigen_sig[:, [5,9,25,21]] # 4 ch raw MIC
            print("Mic shape", mic_sig.shape)
            # Reshape audio signal into frames
            samples_per_frame = fs // 10
            num_frames = mic_sig.shape[0] // samples_per_frame
            print("Number of frames", num_frames)
            assert(num_frames == vsg_sig.shape[1])
            framed_mic_sig = mic_sig[:num_frames * samples_per_frame, :].reshape(num_frames, samples_per_frame, 4)
            print("Framed signal shape", framed_mic_sig.shape)
            mic_data.append(framed_mic_sig) # (nframes, nsamps, nch)
            vg_labels.append(vsg_sig.transpose(1, 0, 2, 3)) # (nframes, nbands, nch, nch)
        a_np = np.vstack(mic_data)
        b_np = np.vstack(vg_labels)

        print("shape of a_np", a_np.shape)
        print("shape of b_np", b_np.shape)

        f.create_dataset("mic", shape=a_np.shape, dtype=a_np.dtype, data=a_np)
        f.create_dataset("labels", shape=b_np.shape, dtype=b_np.dtype, data=b_np)
    
        
        assert(a_np.shape[0] == b_np.shape[0])
        f.attrs["sr"] = fs
        
        del a_np, b_np
        gc.collect()            

create_full_hdf_data(type='train')
# create_full_hdf_data(type='val')