import json
import requests
from pydub import AudioSegment
import torchaudio
import librosa
import soundfile as sf
from Config import Config
import time
from datetime import datetime
import os

"""
TRANSCRIBE_CONFIG = {
    "use_diarization": True, #화자분리
    "diarization": {
        "spk_count": 2
    },
    "use_multi_channel": False,
    "use_itn": True, #단위 변환
    "use_disfluency_filter": True, #간투어 필터
    "use_profanity_filter": False, #비속어 필터
    "use_paragraph_splitter": False, #문단 구분
    }
"""

class MySTT:
    def __init__(self,
                 CLIENT_ID = Config.CLIENT_ID,
                 CLIENT_SECRET = Config.CLIENT_SECRET) :
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET

    def Access_Token(self):
        resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': self.CLIENT_ID,
            'client_secret': self.CLIENT_SECRET}
        )
        resp.raise_for_status()
        access_json = resp.json()
        self.TOKEN = access_json['access_token']
        return access_json
    
    def Post(self, file_path, config) :
        resp = requests.post(
            'https://openapi.vito.ai/v1/transcribe',
            headers={'Authorization': 'bearer '+ self.TOKEN},
            data={'config': json.dumps(config)},
            files={'file': open(file_path, 'rb')}
        )
        resp.raise_for_status()
        TRANSCRIBE_ID = resp.json()['id']

        return TRANSCRIBE_ID
    
    def Transcribe(self, save_name, TRANSCRIBE_ID,
                   json_folder = '/home/alpaco/zenodo/kkw/kkw_json') :
        
        try:
            if not os.path.exists(json_folder):
                os.makedirs(json_folder)
        except OSError:
            print("Error: Failed to create the directory.")

        resp = requests.get(
        'https://openapi.vito.ai/v1/transcribe/'+f'{TRANSCRIBE_ID}',
        headers={'Authorization': 'bearer '+self.TOKEN},
        )
        resp.raise_for_status()

#        print("전사 시작 : ", datetime.now(), TRANSCRIBE_ID)

        try :
            while resp.json()['status'] == 'transcribing':
                time.sleep(5)
                resp = requests.get(
                'https://openapi.vito.ai/v1/transcribe/'+f'{TRANSCRIBE_ID}',
                headers={'Authorization': 'bearer '+self.TOKEN},
                )
                resp.raise_for_status()

            if resp.json()['status'] == 'completed':
                #file_name = file_path.split('/').pop()           
                with open(os.path.join(json_folder, save_name), 'w', encoding = 'UTF-8') as f:
                    json.dump(resp.json()['results'], f, ensure_ascii=False)           
 #               print("전사 완료 : ", datetime.now())   
            
            else :
                print('전사 실패 :', resp.json()['results'])
        except :
            print("유효하지 않은 요청", resp.json())

        return resp.json()

