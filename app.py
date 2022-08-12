
from distutils.command.build_scripts import first_line_re
from fileinput import filename
from math import remainder
from pickle import MEMOIZE, TRUE
from textwrap import indent
from cv2 import FileStorage
from flask import Flask, flash, request, redirect, url_for, render_template,Response
from grpc import method_handlers_generic_handler
from werkzeug.utils import secure_filename
import pandas as pd
import cv2,os
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from tensorflow import keras
import urllib.request
from fileinput import filename
import cv2,os,sys
import numpy as np
from cv2 import CascadeClassifier
from deepface import DeepFace

app = Flask(__name__)

def f1_score(y_true, y_pred): #taken from old keras source code
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + K.epsilon())
     recall = true_positives / (possible_positives + K.epsilon())
     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
     return f1_val
model = tf.keras.models.load_model('./static/emotions.h5',custom_objects={"f1_score": f1_score })
 
UPLOAD_FOLDER = 'static/Uploadfolder'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            #print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed below')

            
            image = cv2.imread(filepath)
            image = cv2.resize(image,(48,48))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image,axis=0)
            # image = cv2.resize(file(48,48))
            # img=np.ndarray(image)
            # print(img.shape)
            # img=img.reshape(1,48,48,1)
            predict_x=model.predict(image) 
            result=np.argmax(predict_x,axis=1)
            labels = ['angry','disgust','fear','happy', 'neutral','Sad','Surprise']
            for i in range(0,7):
                    if result[0]==i:
                        label = labels[i]

            print(result)

            mood_music = pd.read_csv("music_data.csv")
            mood_musics = mood_music[['name','artist','mood','link']]
            mood_musics.set_index(['name'])
            song = ""
            if(result[0]==0 or result[0]==1 or result[0]==2 ):
                #for angery,disgust,fear
                filter1=mood_musics['mood']=='Calm'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=1)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song  = f2
                link = f2['link']
                link.to_frame()
                
            if(result[0]==3 or result[0]==4):
                #for happy, neutral
                filter1=mood_musics['mood']=='Happy'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=1)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song = f2
                link = f2['link']
                link.to_frame()
            if(result[0]==5):
                #for Sad
                filter1=mood_musics['mood']=='Sad'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=1)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song = f2
                link = f2['link']
                link.to_frame()
            if(result[0]==6):
                #for surprise
                filter1=mood_musics['mood']=='Energetic'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=1)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                link = f2['link']
                link.to_frame()
                song = f2
            
            # f2.to_csv('songs.csv')
            return render_template("next.html", name =f2['name'],link =f2['link'],artist = f2['artist'],mood = f2['mood'],titles=[''],label = label)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)         
    return render_template("index.html")

@app.route('/scanner',methods = ['POST','GET'])
def scan():
    if request.method == 'POST':
        if request.form.get('click') == 'Live Scanner':
            return render_template('scanner.html')
    return render_template('index.html')
    
(width, height) = (130, 100)   
face_cascade = cv2.CascadeClassifier('face_detection\Cascade\haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, im = webcam.read()  # read the camera frame
        
        if not success:
            break
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # face = gray[y:y + h, x:x + w]
                # face_resize = cv2.resize(face, (width, height))
                # im = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
                
                img = cv2.resize(im,(48,48))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(gray,axis=0)
                
                labels = ['angry','disgust','fear','happy', 'neutral','Sad','Surprise']
                predict_x=model.predict(img) 
                global result
                result=np.argmax(predict_x,axis=1)
                for i in range(0,7):
                    if result[0]==i:
                        label = labels[i]

                #analysis = DeepFace.analyze(, actions = [ "emotion"])

                cv2.rectangle(im,(x,y-40),(x+w,y),(0,0,0),-1) 
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,0),2) 
                cv2.putText(im,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 

            

            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

            

            key = cv2.waitKey(10)
            if key == 27:
                break





@app.route('/video_feed')  
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/',methods=['POST','GET'])
def song():
    if request.method=='POST':
        if request.form.get('Song') == 'Song':
            gen_frames()
            mood_music = pd.read_csv("music_data.csv")
            mood_musics = mood_music[['name','artist','mood','link']]
            mood_musics.set_index(['name'])
            song = ""
            global f2
            if(result[0]==0 or result[0]==1 or result[0]==2 ):
                #for angery,disgust,fear
                filter1=mood_musics['mood']=='Calm'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=5)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song  = f2
                
            if(result[0]==3 or result[0]==4):
                #for happy, neutral
                filter1=mood_musics['mood']=='Happy'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=5)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song = f2
            if(result[0]==5):
                #for Sad
                filter1=mood_musics['mood']=='Sad'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=5)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song = f2
            if(result[0]==6):
                #for surprise
                filter1=mood_musics['mood']=='Energetic'
                f1=mood_musics.where(filter1)
                f1=f1.dropna()
                f2 =f1.sample(n=5)
                f2.reset_index(inplace=True)
                # del f2['mood'] 
                song = f2
            return render_template("next.html", tables=[f2.to_html()], titles=[''])


    return render_template("scanner.html")

if __name__ == "__main__":
    app.run(debug=True)
