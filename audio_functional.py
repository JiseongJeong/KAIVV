import torchaudio
import soundfile
from pydub import AudioSegment
import librosa

import torch

import numpy as np
import pickle


def load_torch_model(model_class, checkpoint_path, **kwargs):
    checkpoint = torch.load(checkpoint_path)
    model = model_class(**kwargs) # 빈 모델 인스턴스를 선언
    model.load_state_dict(checkpoint['model_state_dict'])
    print('epoch', checkpoint['epoch'], 'val_loss', checkpoint['val_loss'], 'val_accuracy', checkpoint['val_accuracy'])
    return model

def load_ml_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_audio(wav_path):
    ext = wav_path.split('.')[-1].lower()

    if ext in ['wav', 'flac', 'aac']:
        arr, sr = torchaudio.load(wav_path)

    elif ext in ['m4a']:
        sound = AudioSegment.from_file(wav_path, format='m4a')
        file_handle = sound.export('temp.wav', format='wav')
        arr, sr = torchaudio.load(file_handle)

    elif ext in ['pcm']: # pcm_header가 None이면 input()을 받아오도록 만들 수도 있다
        arr, sr = soundfile.read(wav_path, format='RAW', channels=1, samplerate=16000, subtype='PCM_16', endian='LITTLE')
        arr = torch.tensor(arr).unsqueeze(0)

    else:
        raise TypeError('지원하는 오디오 파일 확장자가 아닙니다.')
    
    return arr, sr # type(arr) == torch.Tensor, arr.shape == (channels, sr*duration(sec))


def audio_load_pad_mel_lbrs(wav, crop_from=None, target_sec=4, sample_rate=16000):
    # Calculate target length in samples
    if type(wav) == str:
        wav, _ = librosa.load(wav, sr = 16000)
    target_length_samples = int(target_sec * sample_rate)
    wav = np.array(wav.squeeze())
    # wav = wav[crop_from : crop_from+target_length_samples]
    current_length_samples = wav.shape[-1]
    padding_samples = target_length_samples - current_length_samples
    # Pad the waveform with zeros
    padding_waveform = np.zeros((padding_samples), dtype=np.int32)
    padded_waveform = np.concatenate((wav, padding_waveform))

    spect = librosa.feature.melspectrogram(y=padded_waveform, sr=16000, n_fft=2048, n_mels = 128, win_length=400, hop_length=512)
    data_spec = librosa.power_to_db(spect, ref=np.max)
    mel_spectrogram_np = np.array(data_spec, np.float32)
    mel_spectrogram = torch.Tensor(mel_spectrogram_np)

    return mel_spectrogram # (126, 128)의 텐서