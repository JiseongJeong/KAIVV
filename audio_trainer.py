import pickle
import glob

import torch
from torch.utils.data import Dataset

# import torchaudio
# import soundfile
# from pydub import AudioSegment


def load_dataset(features=['wav_path', 'label', 'crop_from', 'audio_value_sixth', 'audio_value_last']):
    train_dataset, val_dataset, test_dataset = [], [], []
    for tvt in ['train', 'val', 'test']:
        for pkl in glob.glob(f"/home/alpaco/KHR/MM_whole/{tvt}_*.pkl"):
            with open(pkl, 'rb') as f:
                if tvt == 'train':
                    train_dataset += pickle.load(f)
                elif tvt == 'val':
                    val_dataset += pickle.load(f)
                elif tvt == 'test':
                    test_dataset += pickle.load(f)

    # train_dataset = [{feature : data[feature] for feature in features} for data in train_dataset]
    # val_dataset = [{feature : data[feature] for feature in features} for data in val_dataset]
    # test_dataset = [{feature : data[feature] for feature in features} for data in test_dataset]

    train_dataset = [tuple(data[feature] for feature in features) for data in train_dataset]
    val_dataset = [tuple(data[feature] for feature in features) for data in val_dataset]
    test_dataset = [tuple(data[feature] for feature in features) for data in test_dataset]

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    return train_dataset, val_dataset, test_dataset


class CustomDataset(Dataset):
    def __init__(self, data): # data는 샘플(dictionary)들의 리스트
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.data[idx]


def trainer_collate_fn(batch, hubert_layer_num):
    audio_values, labels = [], []
    infos = [] 
    for sample in batch:
        # 오디오 피쳐맵
        audio_values.append(torch.tensor(sample[hubert_layer_num])) # sixth, last 정하기
        # 라벨
        labels.append(sample['label'])
        # 사후분석을 위한 파일 정보
        infos.append(sample['wav_path'])

    # 타입 변환    
    audio_values = torch.stack(audio_values)
    labels = torch.tensor(labels) # .float()은 CrossEntropy 계산할때 해줄거임
    # print(audio_values.shape, type(audio_values))
    # print(labels.shape, type(audio_values))

    return audio_values, labels, infos # 텐서들의 튜플을 반환


class MyAudioClassifier(torch.nn.Module):
    def __init__(self):
        super(MyAudioClassifier, self).__init__()
        self.projector = torch.nn.Linear(in_features=768, out_features=256, bias=True)
        # self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(in_features=256, out_features=2, bias=True)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        proj = self.projector(x)
        # x = self.dropout(x)
        x = self.classifier(proj)
        # x = self.sigmoid(x)
        return proj