"""
UPDATES

09.05
- input type을 지정했습니다.

08.31
-streo > mono 코드를 추가했습니다
-fn_idxs, fp_idxs를 return하도록 만들었습니다.

"""

# 언젠가 requirements로 묶어버릴 것들
import os
from pydub import AudioSegment
import torchaudio
from transformers import HubertModel

import glob
import torch
from tqdm import tqdm
import pickle

from IPython.display import Audio
import soundfile

import numpy as np
from inspect import signature
from time import time, localtime, strftime
from sklearn import metrics

class myhubert :
    def __init__(self, model_name = "facebook/hubert-base-ls960"):
        self.model = HubertModel.from_pretrained(model_name)
        #그래픽 카드 활성화
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

    def make_wav_list(self, folder_path: str, pattern: str = '*') :
        #pattern에는 glob.glob 기준으로 원하는 형식 넣어주기 (없을 시 '*)
        wav_list = glob.glob( folder_path + '/' + pattern)
        return wav_list


    def process_audio(self, 
                    wav_list: list,
                    sample_rate: int, # 원하는 sample_rate
                    second: int,  # 몇 초로 자를지
                    label_condition: str, #label이 1일 조건, str 값으로 넣어줘야함!!!
                    save_path: str = '',
                    save: bool = True) -> (list,list,list) : 
        
        time_id = strftime('%Y%m%d%I', localtime())

        #자료 담을 빈 리스트
        features, labels, cut_audios = [], [], []

        for wav in tqdm(wav_list):
            label = 1 if eval(label_condition) else 0
            input_format = wav.split('.')[-1]
            sound = AudioSegment.from_file(wav, format=input_format)
            if sound.channels != 1 : # channel을 mono로 전환
                sound = sound.set_channels(1) 
            file_handle = sound.export('temp.wav', format='wav')
            arr, sr = torchaudio.load(file_handle)

            if sr != sample_rate:
                arr = torchaudio.functional.resample(arr, sr, sample_rate)
            
            samples = int(sample_rate * second)

            for i in range(0, arr.shape[1]//samples + 1):
                cut = arr[:, samples*i : samples*(i+1)]
                # print(cut.shape)
                cut_audios.append(cut)
                hidden_state = self.model(cut.to("cuda:0"), output_hidden_states=True).hidden_states[6]
                feats = hidden_state.detach().cpu().numpy().squeeze().mean(0)

                features.append(feats)
                labels.append(label)

        if save : 
            with open(os.path.join(save_path, f'features_{time_id}.pkl'), 'wb') as f:
                pickle.dump(features, f)
            with open(os.path.join(save_path, f'labels_{time_id}.pkl'), 'wb') as f:
                pickle.dump(labels, f)
            with open(os.path.join(save_path, f'cut_audios_{time_id}.pkl'), 'wb') as f:
                pickle.dump(cut_audios, f)

        self.features = features
        self.labels = labels
        self.cut_audios = cut_audios

        return features, labels, cut_audios


    def load_classifier(self, model_path: str) :
        if model_path.endswith('pkl') :
            with open(model_path,'rb') as f:
                classifier = pickle.load(f)
        else :
            print(".pkl만을 지원합니다.")

        return classifier
        

    def reports(self, classifier, x, y_true) -> (np.array,np.array) :
        y_pred = classifier.predict(x)

        # fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        # fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

        print("Classification Report :")
        print(metrics.classification_report(y_true, y_pred))
        print("=" * 50)
        # print("Indices of False Positives:", fp_indices)
        # print("Indices of False Negatives:", fn_indices)

        print('실제 0인 인덱스 : ', np.where(y_true == 0)[0])
        print('실제 1인 인덱스 : ', np.where(y_true == 1)[0])
        print("=" * 50)

        zero_idxs = np.where(y_pred == 0)[0]
        one_idxs = np.where(y_pred == 1)[0]
        true_zeros = y_true[zero_idxs]
        true_ones = y_true[one_idxs]
        print('0으로 예측한 인덱스 :', zero_idxs)
        print('            실제값 :', true_zeros)
        print()
        print('  1로 예측한 인덱스 :', one_idxs)
        print('            실제값 :', true_ones)
        print("=" * 50)

        fn_msk = np.ma.masked_array(zero_idxs, mask=true_zeros)
        fn_idxs = fn_msk[fn_msk.mask == True].data
        print('1인데 0으로 예측한 인덱스(FN) : ', fn_idxs)

        fp_msk = np.ma.masked_array(one_idxs, mask=true_ones)
        fp_idxs = fp_msk[fp_msk.mask == False].data
        print('  0인데 1로 예측한 인덱스(FP) : ', fp_idxs)

        return fn_idxs, fp_idxs

