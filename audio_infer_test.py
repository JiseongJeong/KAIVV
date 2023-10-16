import torchaudio
import time



def run(audio_path):
    arr, sr = torchaudio.load(audio_path)
    global text1, text2, text3, text4, progress
    text1, text2, text3, text4, progress = "", "", "", "", "loading...(0%)"
    #time.sleep(2)  # Pause for 2 seconds
    text1 = f'{arr.size(1)/sr}초짜리 음성이군여..'
    #progress = "loading...(25%)"
    print(progress)
    #time.sleep(2)
    text2 = f'샘플레이트는 {sr}이군요...'
    #progress = "loading...(50%)"
    #print(progress)
    #time.sleep(2)
    text3 = str(arr)
    #progress = "loading...(75%)"
    #print(progress)
    #time.sleep(2)
    text4 = '아무튼 저는 오디오 파일을 잘 읽어왔습니다.'
    #progress = "loading...(100%)"
    #print(progress)
    return text1 + '\n' + text2 + '\n' +  text3 + '\n' +  text4
#run('static/audio/gen_100.wav')