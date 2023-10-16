from audio_functional import *
from transformers import HubertModel
from audio_trainer import MyAudioClassifier
from JJS_custom_models import MK1_SimpleCNN_snatch

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

# final model의 구조를 담은 모델 클래스 추가할 것
class MultiModal_mk1(torch.nn.Module):
    def __init__(self, image_model_path, audio_model_path):
        super(MultiModal_mk1, self).__init__()
        self.fc1 = torch.nn.Linear(384, 64)
        self.bn = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()
        # self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, 2)  # binary

        self.image_model = load_torch_model(MK1_SimpleCNN_snatch, image_model_path)
        self.audio_model = load_torch_model(MyAudioClassifier, audio_model_path)

    def forward(self, mels, audio_value_lasts):
        image_feat = self.image_model(mels) # (128,) # forward가 snatch 부분만 반환하는 것 확인했음
        audio_feat1 = self.audio_model(audio_value_lasts) # (256,)

        x = torch.cat([image_feat, audio_feat1], dim=-1)

        x = self.fc1(x)
        x = self.bn(x)
        # x = self.gelu(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def preprocess_audio(wav_path, sampling_rate=16000, crop_duration=4, overlap=2):
    hz = crop_duration * sampling_rate
    arr, sr = load_audio(wav_path)

    # 스테레오라면 모노로
    if arr.size(0) == 2:
        arr = torch.mean(arr, dim=0).unsqueeze(0)

    # 리샘플링
    if sr != sampling_rate:
        arr = torchaudio.functional.resample(arr, sr, sampling_rate)

    # 오버래핑하여 크롭
    audio_crops = []
    for i in range(0, max(1, round(arr.shape[1] / hz))*hz, hz - overlap*sampling_rate):
        audio_crop = arr[:, i : i + hz]
        # 패딩
        audio_crop = torch.nn.functional.pad(audio_crop, (0, hz - audio_crop.size(1)), "constant", 0) # .squeeze()
        audio_crops.append(audio_crop)
    return audio_crops


def crop_to_data(audio_crop, hubert_model):
    assert audio_crop.size(1) == 64000, 'input must be A cropped audio in tensor with a length of 64000'

    mels = audio_load_pad_mel_lbrs(audio_crop, crop_from=0)

    audio_value = hubert_model(audio_crop.to(device), output_hidden_states=True)
    audio_value_sixths = audio_value.hidden_states[6].squeeze().mean(0) # .detach().cpu().numpy()
    audio_value_lasts = audio_value.last_hidden_state.squeeze().mean(0)

    return mels.unsqueeze(0), audio_value_sixths.unsqueeze(0), audio_value_lasts.unsqueeze(0)


def crops_to_data(audio_crops): # 배치단위로
    audio_crops = torch.stack(audio_crops)
    return


def load_hubert(hubert_name="facebook/hubert-base-ls960", device=device):
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    print(f'[{hubert_name} loaded]')
    hubert = hubert.to(device)
    print('...............to', device)
    return hubert



