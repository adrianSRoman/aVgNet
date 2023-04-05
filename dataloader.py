import h5py
import librosa
import torch.utils.data as data
import numpy as np

class HDF5AudioDataset(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.audio_data = self.file['mic'] # MIC audio data (nframes, nsamps, nch)
        self.label_data = self.file['labels'] # visibility graph matrices (nframes, nbands, nch, nch)

    def __getitem__(self, index):
        '''
        Ouput: complex-valued stft 
        '''
        audio_frame = self.audio_data[index] # 100msec audio frames
        frame_stft = np.array([librosa.stft(audio_frame[:, i], n_fft=1024, hop_length=64) for i in range(audio_frame.shape[1])])
        label = self.label_data[index]
        data = np.stack((frame_stft.real, frame_stft.imag), axis=0) # (batch_sz, 2, nch, fftbins, n_frames)
        label = np.stack((label.real, label.imag), axis=0)
        return data, label

    def __len__(self):
        return len(self.audio_data)

def get_data_loader(file_path, batch_size=32, shuffle=True):
    dataset = HDF5AudioDataset(file_path)
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
