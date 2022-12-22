from flask import Flask, render_template, Response
# from score_test2 import action_answer
import time
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np

from user_answer import action_answer
from sep_song import song_by_song
from Score import scoring_answer


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.js')

def gen(song_num,level='easy'):
    actions = ['rabbit', 'mountain', 'go', 'santa', 'twinkle', 'nose', 'butterfly', 'flower', 'bird', 'bear','fat', 'thin', 'cute']
    seq_length = 30

    model = load_model('model.h5')

    correct_actions, correct_act_ko, cut_time, stroke_fill = song_by_song(song_num,level)
    
    max_time = time.time() + cut_time[-1]  # 종료 시간 설정
    cut_time_list = [] 
    c_time = time.time() # 시작 시간
    # 동작 구분 시간
    for i in range(len(cut_time)) :
        t = c_time + cut_time[i]
        cut_time_list.append(t)
    answer = action_answer(model,actions,seq_length,correct_actions,correct_act_ko,cut_time_list,stroke_fill)
    # 카메라 켜기
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        img, labels = answer.answer(cap)

        # out.write(img0)
        # out2.write(img)
        ret, frame = cv2.imencode('.jpg', img)
        # 설정한 종료 시간이 되면 while 문 탈출
        if time.time() > max_time :
                    break
        if cv2.waitKey(1) == ord('q'):
            break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')

    label = labels
    test = scoring_answer(actions, correct_actions, label, cut_time_list,level)
    score, wrong_action = test.score()

    print(score)
    print(wrong_action)

@app.route('/video_feed/<int:animalId>/<level>')
def video_feed(animalId, level):
    return Response(gen(str(animalId),level),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True, use_reloader=False)