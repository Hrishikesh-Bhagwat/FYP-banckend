import json
import os
os.system("sudo apt-get install libsndfile1")
import re
import stat
from flask import Flask, jsonify,request,render_template, send_from_directory,send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pickle
import soundfile
import matplotlib.pyplot as plt
import uuid
import numpy as np
import librosa.display
import librosa

Pkl_Filename = "Emotion_Voice_Detection_Model.pkl"
Emotion_Voice_Detection_Model=None
with open(Pkl_Filename, 'rb') as file:  
    Emotion_Voice_Detection_Model = pickle.load(file)

app = Flask(__name__)
cors = CORS(app)

#one endpoint to load page
#one endpoint to accept file input
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

@app.route('/')
def hello_world():
   return send_from_directory(path='./templates')

@app.route('/analyze',methods=["POST"])
def analyze_audio():
   x=str(uuid.uuid4())
   print(x)
   print(type((json.loads(request.data))["bytes"][0]))
   with open(f'{x}.wav', mode='bx') as f:
        f.write(bytes((json.loads(request.data))["bytes"]))
   data , sr = librosa.load(x+'.wav')
   data = np.array(data)
   plt.figure(figsize=(15, 5))
   librosa.display.waveshow(data, sr=sr)
   plt.savefig(x+"_plot"+".png")
   ans=[]
   new_feature = extract_feature(f'{x}.wav', mfcc=True, chroma=True, mel=True)
   ans.append(new_feature)
   ans = np.array(ans)
   result=Emotion_Voice_Detection_Model.predict(ans)[0]
   return jsonify(emotion=result,file_plot=x+"_plot.png",file_audio=x+".wav")

@app.route('/files',methods=['GET'])
def send_image():
    return send_file(request.args.get("name"))
if __name__ == '__main__':
   app.run(debug=True,host="localhost")