from STTVITO import MySTT
from MyPolyglot import *
from transformers import BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import re

TRANSCRIBE_CONFIG = {
    "use_diarization": False, #화자분리
    # "diarization": {
    #     "spk_count": 2
    # },
    "use_multi_channel": False,
    "use_itn": True, #단위 변환
    "use_disfluency_filter": True, #간투어 필터
    "use_profanity_filter": False, #비속어 필터
    "use_paragraph_splitter": True,
    "paragraph_splitter": {
    "max": 1 #문단 구분
    }
    }

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class InferenceNLP:
    def __init__(self, sentclf_id=None, docclf_id=None):
        stt = MySTT(CLIENT_ID='PMu5GOTqHHbqZhlTIedR',
            CLIENT_SECRET= 'Bpx7neHTADCUv35mtFDARKzHuD0tKifg2JirHmUC')
        access_json = stt.Access_Token()
        self.stt = stt
        self.access_json = access_json
        if sentclf_id : 
            self.sent_clf = PolyglotForInfer()
            self.sent_clf.load_model(sentclf_id)
        if docclf_id : 
            self.doc_clf = PolyglotForInfer()
            self.doc_clf.load_model(docclf_id)


    def ToText(self, file_path) :
        file_name = file_path.split('/')[-1]
        save_name = file_name.split('.')[0] + '.json'
        TRANSCRIBE_ID = self.stt.Post(file_path, config = TRANSCRIBE_CONFIG)
        output = self.stt.Transcribe(save_name, TRANSCRIBE_ID)
        utters = output['results']['utterances']
        return utters
    
    def SentenceClassification(self, utters) :
        results = []
        with torch.no_grad():
            for utter in utters :
                msg = utter['msg']
                if len(msg) < 5 :
                    continue
                msg = re.sub('[a-zA-Z0-9]', '', msg)
                input = self.sent_clf.tokenizer(msg, return_tensors="pt")
                output = self.sent_clf.model(input['input_ids'].to('cuda')).logits[0]
                prob = torch.sigmoid(output).cpu().numpy()
                result = {'start_at' : utter['start_at'],
                        'duration' : utter['duration'],
                        'msg' : utter['msg'],
                        'normal proba' : prob[0],
                        'phishing proba' : prob[1]}
                results.append(result)
        
        return results

    def DocumentClassification(self, utters) :
        with torch.no_grad():
            dialog = ''
            for utter in utters :
                msg = utter['msg']
                dialog += msg + ' '
            input = self.doc_clf.tokenizer(dialog, return_tensors="pt")
            output = self.doc_clf.model(input['input_ids'].to('cuda')).logits[0]
            prob = torch.sigmoid(output).cpu().numpy()
            result = {'dialog' : dialog,
                      'normal proba' : prob[0],
                      'phishing proba' : prob[1]}
            
        return result

    def run(self, file_path) :
        utters = self.ToText(file_path)
        result_sent = self.SentenceClassification(utters)
        result_doc = self.DocumentClassification(utters)

        return result_sent, result_doc
