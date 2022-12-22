import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

class action_answer() :
    def __init__(self,model,actions,seq_length,correct_actions,correct_act_ko,cut_time):
        self.model = model  # model
        self.actions = actions # 총 action
        self.seq_length = seq_length # 인식되는 sequence 길이
        self.correct_actions = correct_actions # 정답 action
        self.correct_act_ko = correct_act_ko # 정답 action 한국어 버전
        self.cut_time = cut_time # action 실행 구간
        self.test_time = cut_time[-1] # 총 영상 길이
    
    # 실시간 action 실행 구간 list 생성
    def timecut(self) : 
        c_time = time.time()
        cut_time_list = []
        for i in range(len(self.cut_time)) :
            t = c_time + self.cut_time[i]
            cut_time_list.append(t)
        
        return cut_time_list

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
        ret, img = cap.read()
        img0 = img.copy()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 특정 시간에 행동 
        for i in range(len(self.correct_actions)):
            if (time.time() < cut_time_list[i+1]) & (time.time() >= cut_time_list[i])  :
                do_action = 'Now ' + self.correct_actions[i] + ' motion!'
                # do_action = 'Now ' + self.correct_actions[i] + ' motion!'
                cv2.putText(img, f'{do_action.upper()}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

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

                if len(seq) < self.seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-self.seq_length:], dtype=np.float32), axis=0)

                y_pred = self.model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = self.actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                if this_action in self.correct_actions:
                    c_time = time.time()
                    cut_off = [i for i in range(len(cut_time_list)-1) if (c_time>=cut_time_list[i])&(c_time<cut_time_list[i+1])]
                    if len(labels) == 0 :
                        labels.append([this_action,time.time(),cut_off])
                    else : 
                        if this_action != labels[-1][0]:
                            labels.append([this_action,time.time(),cut_off]) 
                        else :
                            if cut_off != labels[-1][-1]:
                                labels.append([this_action,time.time(),cut_off]) 
            # cv2.imshow('img', img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return  jpeg.tobytes()

    def get_frame(self):
        ret, frame = self.video.read()
        return jpeg.tobytes()

class Score_test():
    def __init__(self, actions, correct_actions, labels, cut_time_list) :
        self.correct_actions = correct_actions
        self.labels = labels[:-1]
        self.cut_time_list = cut_time_list 
    
    def score(self,level = 'easy') :
        wrong_action = []
        score = 0
        num_correct = len(self.correct_actions)
        if len(self.labels) < num_correct :
            for n in range(num_correct-len(self.labels)) :
                self.labels.append([0,0])
        labels = np.array(self.labels,dtype=object)
        print(labels)
        labels = labels[:,:-1]
        if level == 'easy' :
            for i in range(num_correct) :
                if correct_actions[i] == labels[i,0] :
                    if (float(labels[i][1])>=self.cut_time_list[i]) & (float(labels[i][1])<=self.cut_time_list[i+1]) :
                        score += 1
                    else : 
                        if i == len(labels) :
                            wrong_action.append(correct_actions[i]) 
                        else :
                            if correct_actions[i] in labels[i+1:,0] :
                                idx = np.where(labels == correct_actions[i])
                                k = 0
                                for j in idx[0]:
                                    if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i+1]) :
                                        score += 1
                                        k += 1
                                if k == 0 :
                                    wrong_action.append(correct_actions[i])
                            else :
                                wrong_action.append(correct_actions[i])  
                else : 
                    if correct_actions[i] in labels[:,0] :
                        idx = np.where(labels == correct_actions[i])
                        k = 0
                        for j in idx[0]:
                            if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i+1]) :
                                score += 1
                                k += 1
                        if k == 0 :
                            wrong_action.append(correct_actions[i])
                    else :
                        wrong_action.append(correct_actions[i])
        elif level == 'hard' :
            for i in range(num_correct) :
                if correct_actions[i] == labels[i,0] :
                    if (float(labels[i][1])>=self.cut_time_list[i]) & (float(labels[i][1])<=self.cut_time_list[i+1]) :
                        score += 1
                    else : 
                        if i == len(labels) :
                            wrong_action.append(correct_actions[i]) 
                        else :
                            if correct_actions[i] in labels[i+1:,0] :
                                idx = np.where(labels == correct_actions[i])
                                k = 0
                                for j in idx[0]:
                                    if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i]+3) :
                                        score += 1
                                        k += 1
                                if k == 0 :
                                    wrong_action.append(correct_actions[i])
                            else :
                                wrong_action.append(correct_actions[i])  
                else : 
                    if correct_actions[i] in labels[:,0] :
                        idx = np.where(labels == correct_actions[i])
                        k = 0
                        for j in idx[0]:
                            if (float(labels[j][1])>=self.cut_time_list[i]) & (float(labels[j][1])<=self.cut_time_list[i]+3) :
                                score += 1
                                k += 1
                        if k == 0 :
                            wrong_action.append(correct_actions[i])
                    else :
                        wrong_action.append(correct_actions[i])
            else : 
                print('Input correct level')
        score /= num_correct
        return score, wrong_action

# actions = ['rabbit', 'mountain', 'go','snata','snow', 'nose',
#                 'butterfly','flower', 'bird','bear','fat', 'thin','cute']
# correct_actions = ['bear','fat', 'thin','cute']
# correct_act_ko = ['곰','뚱뚱한','날씬한','귀여운']
# seq_length = 30
# test_time = 20
# cut_time_original = [0,6,13,20,29] # [0,4,10,20]
# plus_list = [6] * len(cut_time_original)
# cut_time = [a+b for a,b in zip(cut_time_original,plus_list)]
# model = load_model('model.h5')

# socre test 실행
# answer = action_answer(model,actions,seq_length,correct_actions,correct_act_ko,cut_time)
# frames, labels, cut_time_list = answer.answer()
# print(labels)
# print(cut_time_list)
# test = Score_test(actions, correct_actions, labels, cut_time_list)
# score, wrong_action = test.score(level='hard')
# print(score)
# print(wrong_action)

