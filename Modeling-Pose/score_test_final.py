import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from PIL import ImageFont, ImageDraw, Image

class action_answer() :
    def __init__(self,model,actions,seq_length,correct_actions,correct_act_ko,cut_time_original):
        self.model = model  # model
        self.actions = actions # 총 action
        self.seq_length = seq_length # 인식되는 sequence 길이
        self.correct_actions = correct_actions # 정답 action
        self.correct_act_ko = correct_act_ko # 정답 action 한국어 버전

        plus_list = [6] * len(cut_time_original) # 카메라 delay 시간
        self.cut_time = [a+b for a,b in zip(cut_time_original,plus_list)] # action 실행 구간
        self.test_time = self.cut_time[-1] # 총 영상 길이

    # 유저의 동작을 list 형태로 보냄
    def answer(self) :
        # 손 모양 캐치
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        seq = []
        action_seq = []
        labels = []
        
        max_time = time.time() + self.test_time  # 종료 시간 설정
        cut_time = self.cut_time
        cut_time_list = [] 
        c_time = time.time() # 시작 시간
        # 동작 구분 시간
        for i in range(len(cut_time)) :
            t = c_time + cut_time[i]
            cut_time_list.append(t)
        # 카메라 켜기
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            img0 = img.copy()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            fontpath = "fonts/gulim.ttc"
            font = ImageFont.truetype(fontpath, 35)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            # 특정 시간에 행동 명령어 출력
            for i in range(len(self.correct_actions)):
                if (time.time() < cut_time_list[i+1]) & (time.time() >= cut_time_list[i])  :
                    # do_action = 'Now ' + self.correct_actions[i] + ' motion!' # 영어 버전
                    do_action = '이제, ' + self.correct_act_ko[i] + ' 동작 해봐요!' # 한국어 버전
                    draw.text((60, 30), f'{do_action.upper()}', font=font, fill=(255, 255, 255, 0))
                    img = np.array(img_pil)
                    # cv2.putText(img, f'{do_action.upper()}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            # 손의 landmark 정보를 통해 사용자의 행동 정보 list 추출
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if len(seq) < self.seq_length: # 설정한 시퀀스 킬이보다 클 경우만 이후 명령어 실행
                        continue

                    input_data = np.expand_dims(np.array(seq[-self.seq_length:], dtype=np.float32), axis=0)

                    y_pred = self.model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9: # 라벨에 대한 최대 예측 점수가 0.9가 넘어야 하는 조건
                        continue

                    action = self.actions[i_pred]
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    this_action = '?'
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]: # 연속된 3개의 action_seq가 들어와야 알맞는 동작으로 인식
                        this_action = action

                    # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    if this_action in self.correct_actions: # 출련된 action이 동요에 알맞는 action 인지 확인
                        c_time = time.time() 
                        # 동작 구분 시간을 구함으로써 중복 제거 방지
                        cut_off = [i for i in range(len(cut_time_list)-1) if (c_time>=cut_time_list[i])&(c_time<cut_time_list[i+1])]
                        if len(labels) == 0 :
                            labels.append([this_action,time.time(),cut_off])
                        else : 
                            if this_action != labels[-1][0]: # 전 동작과 같은 동작을 반복해서 인식하는 경우 하나의 동작 실행으로 받아들임
                                labels.append([this_action,time.time(),cut_off]) 
                            else :
                                if cut_off != labels[-1][-1]: # 반복하지만 동작 구분 시간이 다른 경우 다른 동작의 실행으로 받아들임
                                    labels.append([this_action,time.time(),cut_off]) 
            cv2.imshow('img', img)
            # 설정한 종료 시간이 되면 while 문 탈출
            if time.time() > max_time :
                        break
            if cv2.waitKey(1) == ord('q'):
                break
        
        return  labels, cut_time_list

# 사용자 동작 결과에 대한 score 산출
class Score_test():
    def __init__(self, actions, correct_actions, labels, cut_time_list) :
        self.correct_actions = correct_actions # 정답 action
        self.labels = labels[:-1] # 사용자 동작 결과
        self.cut_time_list = cut_time_list # 동작 구분 시간
    
    def score(self,level = 'easy') : # 레벨에 따른 다른 score 산출 방식
        if len(self.labels) == 0 :
            score =  0
            wrong_action = self.correct_actions
            return score, wrong_action
        wrong_action = []
        score = 0
        num_correct = len(self.correct_actions)
        # 사용자의 동작 결과가 정답보다 적을 경우 남은 갯수만큼 빈 list로 채우기(out of index를 방지하기 위함)
        if len(self.labels) < num_correct :
            for n in range(num_correct-len(self.labels)) :
                self.labels.append([0,0])
        labels = np.array(self.labels,dtype=object)
        print(labels)
        labels = labels[:,:-1] # 동작 구분 시간 결과값 떼어내기

        if level == 'easy' :
            # 정답을 기준으로 사용자 행동 결과를 하나씩 비교
            for i in range(num_correct) :
                # 정답과 같은 위치에 사용자 동작 결과가 들어왔는지 확인
                if self.correct_actions[i] == labels[i,0] : 
                    # 해당 동작이 정확한 동작 구분 시간에 실행했으면 맞다고 처리
                    if (float(labels[i][1])>=self.cut_time_list[i]) & (float(labels[i][1])<=self.cut_time_list[i+1]) :
                        score += 1
                    else : 
                        # 제일 마지막 동작이 동작 구분시간 안에 없으면 틀림(out of index 방지)
                        if i == len(labels) :
                            wrong_action.append(self.correct_actions[i]) 
                        else :
                            # 중간에 오답이 있어 올바른 행동의 결과가 정답과 다른 위치에서 있는 것을 확인하고 맞다고 처리
                            if self.correct_actions[i] in labels[i+1:,0] :
                                idx = np.where(labels == self.correct_actions[i])
                                k = 0
                                for j in idx[0]:
                                    if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i+1]) :
                                        score += 1
                                        k += 1
                                # 사용자 동작 결과에서 올바른 동작의 동작 구분시간의 범위에 없는 경우
                                if k == 0 :
                                    wrong_action.append(self.correct_actions[i])
                            # 오답인 행동과 같은 행동이 뒷 index에 없는 경우
                            else :
                                wrong_action.append(self.correct_actions[i])  
                else : 
                    # 정답과 다른 위치에 사용자 동작 결과가 있는지 확인
                    if self.correct_actions[i] in labels[:,0] :
                        idx = np.where(labels == self.correct_actions[i])
                        k = 0
                        for j in idx[0]:
                            if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i+1]) :
                                score += 1
                                k += 1
                        if k == 0 :
                            wrong_action.append(self.correct_actions[i])
                    else :
                        wrong_action.append(self.correct_actions[i])
        elif level == 'hard' :
            # 'easy' mode와 실행 원리는 같음 (동작 구분 시작의 구간이 더욱 짧아진 것만 다름)
            for i in range(num_correct) :
                if self.correct_actions[i] == labels[i,0] :
                    if (float(labels[i][1])>=self.cut_time_list[i]) & (float(labels[i][1])<=self.cut_time_list[i+1]) :
                        score += 1
                    else : 
                        if i == len(labels) :
                            wrong_action.append(self.correct_actions[i]) 
                        else :
                            if self.correct_actions[i] in labels[i+1:,0] :
                                idx = np.where(labels == self.correct_actions[i])
                                k = 0
                                for j in idx[0]:
                                    if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i]+3) :
                                        score += 1
                                        k += 1
                                if k == 0 :
                                    wrong_action.append(self.correct_actions[i])
                            else :
                                wrong_action.append(self.correct_actions[i])  
                else : 
                    if self.correct_actions[i] in labels[:,0] :
                        idx = np.where(labels == self.correct_actions[i])
                        k = 0
                        for j in idx[0]:
                            if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i]+3) :
                                score += 1
                                k += 1
                        if k == 0 :
                            wrong_action.append(self.correct_actions[i])
                    else :
                        wrong_action.append(self.correct_actions[i])
            else : 
                print('Input correct level')
        score /= num_correct
        return score, wrong_action
     
# 설정한 action
actions = ['rabbit', 'mountain', 'go','santa','snow', 'nose',
                'butterfly','flower', 'bird','bear','fat', 'thin','cute']
# 1 : 루돌프 , 2 : 곰 세마리, 3 : 산토끼, 4 : 나비야
# 동요의 action
correct_actions_3= ['rabbit','go', 'mountain']
correct_actions_1 = ['twinkle', 'nose','santa']
correct_actions_4 = ['butterfly','flower', 'bird']
correct_actions_2 = ['bear','fat', 'thin','cute']
# action 한국어 버전
correct_act_ko_3 = ['토끼','간다','산']
correct_act_ko_1 = ['반짝','코','산타']
correct_act_ko_4 = ['나비','꽃','참새']
correct_act_ko_2 = ['곰','뚱뚱해','날씬해','귀여워']

seq_length = 30
# 동요별 cut_time
cut_time_3 = [0,4,10,20]
cut_time_1 = [0,4,10,20]
cut_time_4 = [0,4,10,20]
cut_time_2 = [0,6,13,20,29] 

# 모델
model = load_model('models/model.h5')

# score test 실행
song_num = str(input('Input your song number : ')) # 노래 선택
level = str(input('Input your level : ')) # 난이도 선택
# 노래별로 test
def test_by_song(song_num,level='easy'):
    # 노래에 따른 변수 생성
    correct_actions = globals()['correct_actions_' + song_num]
    correct_act_ko = globals()['correct_act_ko_' + song_num]
    cut_time = globals()['cut_time_' + song_num]
    
    # 노래에 따른 test 실행 
    answer = action_answer(model,actions,seq_length,correct_actions,correct_act_ko,cut_time)
    labels, cut_time_list = answer.answer()
    test = Score_test(actions, correct_actions, labels, cut_time_list)
    score, wrong_action = test.score(level=level)

    print(score)
    print(wrong_action)

test_by_song(song_num,level)