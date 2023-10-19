import os
from flask import request, Flask, render_template, redirect, url_for
from InferenceNLP import *
from inference_AUDIO import *
import torch
import pymysql
import json

def connect_to_mysql():
    return pymysql.connect(
        host='localhost',
        port = 3306,
        user='root',
        password='1234',
        database='kaivv_infer_results',
        charset = 'utf8'
    )


def baak(audio_result, result_sent, result_do, script_made_up = None):   # 딕셔너리 출력값을 (상대적)예쁜 텍스트로 출력
    scripts = []
    script = ""
    if script_made_up:
        output = script_made_up
    else:
        output = {}
        for idx in range(len(audio_result)+1):
            if idx == 0:
                output[0] = {'deepfake' : "...", 'msg' : [], 'fraud' : []}  # 처음 0~2초는 일단 정상음성이라고 한다...
            elif idx != len(audio_result):
                output[idx*2] = {'deepfake' : f"위조음성({torch.sigmoid(torch.tensor(audio_result[idx][0][1]))*100:.1f}%)🚨" if audio_result[idx][0][0] < audio_result[idx][0][1] else f"정상음성({torch.sigmoid(torch.tensor(audio_result[idx][0][0]))*100:.1f}%)👌", 'msg' : [], 'fraud' : []}
            else:
                output[idx*2] = {'deepfake' :output[(idx-1)*2]['deepfake'], 'msg' : [], 'fraud' : []} 
                # 마지막은 우리... 2초미만 버렸기때문에 자연어랑 짝이 안맞을 수가 있다. 그래서 그냥 바로 전 타임스탭 값(idx-1)을 띄운다
        for sent in result_sent:
            eos = sent['start_at'] + sent['duration']
            index = (eos//2000)*2
            output[index]['msg'].append(sent['msg'])
            output[index]['fraud'].append(f"fraud ({sent['phishing proba']*100:.1f}%)" if sent['normal proba'] < sent["phishing proba"] else f'normal ({sent["normal proba"]*100:.1f}%)') 

    for idx in output:
        script = "{timestep}\n{deepfake}\n{sents}".format(
            timestep = "🤔탐지중 : " + str(idx) + "~" + str(idx+2) + "초",
            deepfake = output[idx]['deepfake'],
            sents = '\n'.join([i[1]+'\n'+i[0] for i in list(zip(output[idx]['msg'], output[idx]['fraud']))])
            )
        scripts.append(script)
    scripts.append("🤔 통화 전체 내용 \n분석 결과\n : {}".format(
        f"사기 통화 가능성\n\" 높음 \" ({result_doc['phishing proba']*100:.1f})" if result_doc['phishing proba'] > result_doc['normal proba'] else f"정상 통화 가능성\n \" 높음 \"({result_doc['normal proba']*100:.1f})"
    ))
    return scripts

#10.18. 지성 추가 from
def baak_sql(audio_result, result_sent, result_doc):  
    result_sent_json = {}
    audio_result_json = {}

    for i in range(len(audio_result)):
        audio_result_json[i*2000] = 'Fake' if audio_result[i] == 1 else 'Normal'
    for i in range(len(result_sent)):
        key = (result_sent[i]['start_at'] // 2000) * 2000
        if key in result_sent_json:
            result_sent_json[key].append(result_sent[i])
        else:
            result_sent_json[key] = [result_sent[i]]

    return audio_result_json, result_sent_json, result_doc_json



def insert_audio_result(audio_result_json, file_dir):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    cursor.execute("INSERT INTO RESULT (FILE_DIR, RESULT_AUDIO) VALUES (%s, %s)", (file_dir, json.dumps(audio_result_json)))

    connection.commit()
    cursor.close()
    connection.close()


def insert_result_sent(result_sent_json, file_dir):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    # Insert result_sent_json into RESULT table
    cursor.execute("UPDATE RESULT SET RESULT_SENTENCE = %s WHERE FILE_DIR = %s", (json.dumps(result_sent_json), file_dir))

    connection.commit()
    cursor.close()
    connection.close()


def insert_result_doc(result_doc_json, file_dir):
    connection = connect_to_mysql()
    cursor = connection.cursor()

    # Insert result_doc_json into RESULT table
    cursor.execute("UPDATE RESULT SET RESULT_STT = %s WHERE FILE_DIR = %s", (result_doc_json['dialog'], file_dir))

    connection.commit()
    cursor.close()
    connection.close()
#10.18. 지성 추가 end
####################################################################################################################################################################################

app = Flask(__name__, static_url_path= '/static')





UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'audio')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def first_page():
    return render_template('first_page.html')


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['file']

    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        print('if문 입장')
        global result_doc, result_sent, audio_result, scripts, audio_file

        audio_file = app.config['UPLOAD_FOLDER']+'/'+file.filename
        result_sent, result_doc = inlp.run(audio_file)
        wavs = preprocess_audio(audio_file)
        audio_result = []
        for wav in wavs:
            mels, audio_value_sixths, audio_value_lasts = crop_to_data(wav, HUBERT)
            logit = mm_model(mels.unsqueeze(0).to(device), audio_value_lasts.to(device)).detach().cpu().numpy() # audio_value_sixths, # in shape of (1, 2)
            prediction = np.argmax(logit, axis=1)[0] # 라벨을 뱉을거면 prediction, 로짓을 뱉을거면 logit
            audio_result.append(logit) # 라벨을 뱉을거면 prediction, 로짓을 뱉을거면 logit
            
        scripts = baak(audio_result, result_sent, result_doc)


        #10.18. 지성 추가 from
        audio_result_json, result_sent_json, result_doc_json = baak_sql(audio_result, result_sent, result_doc)
        insert_audio_result(audio_result_json, audio_file)
        insert_result_sent(result_sent_json, audio_file)
        insert_result_doc(result_doc_json, audio_file)
        #10.18. 지성 추가 end

        return redirect(url_for('main_page'))
    else:
        return redirect(url_for('main_page'))
@app.route("/main")
def main_page():
    return render_template('main_page.html')


@app.route("/result")
def result_page():
    return render_template('result.html', scripts=scripts, audio_file = audio_file)

@app.route("/phone")
def phone_page():
    return render_template('phone_templates.html')

# @app.route("/test")
# def test_page():
#     return render_template('test.html')


if __name__ == "__main__":
    print('모듈 가져오기 완료')
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    image_model_path = '/home/alpaco/mel_spec_approach/khrdatatest/mk1_khrdatatest_lr0001_12.pt'
    audio_model_path = '/home/alpaco/KHR/MM_whole/checkpoints/new_my_audio_classifier_last_088_88.pt'
    # mm_model = MultiModal_mk1(image_model_path, audio_model_path).to(device)
    final_model_class = MultiModal_mk1
    final_checkpoint_path = '/home/alpaco/KHR/MM_whole/mm_checkpoints/mm_focal_4.pt'
    mm_model = load_torch_model(final_model_class, final_checkpoint_path, image_model_path=image_model_path, audio_model_path=audio_model_path).to(device)
    print('멀티모달 로딩 완료')
    mm_model.eval()
    HUBERT = load_hubert()
    print('휴버트 로딩 완료')

    inlp = InferenceNLP(
        sentclf_id = '/home/alpaco/csy/1005_sent_clf_with_clean',
        docclf_id = '/home/alpaco/csy/polyglot_for_document_clf_0926'
    )
    print('nlp모델 로딩 완료')    
    app.run(debug=False)