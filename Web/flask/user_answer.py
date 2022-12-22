import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time
from PIL import ImageFont, ImageDraw, Image

class action_answer() :
    def __init__(self,model,actions,seq_length,correct_actions,correct_act_ko,cut_time_list,stroke_fill):
        self.model = model  # model
        self.actions = actions # 총 action
        self.seq_length = seq_length # 인식되는 sequence 길이
        self.correct_actions = correct_actions # 정답 action
        self.correct_act_ko = correct_act_ko # 정답 action 한국어 버전
        self.fontpath = "BMJUA_ttf.ttf"

        self.cut_time_list = cut_time_list
        self.test_time = self.cut_time_list[-1] # 총 영상 길이
        self.stroke_fill = stroke_fill

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.action_seq = []
        self.seq = []
        self.labels = []

    # 유저의 동작을 list 형태로 보냄
    def answer(self,cap) :
        ret, img = cap.read()
        img0 = img.copy()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        font = ImageFont.truetype(self.fontpath, 50)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        # 특정 시간에 행동 명령어 출력
        for i in range(len(self.correct_actions)):   
            if (time.time() < min(self.cut_time_list[i+1],self.cut_time_list[i]+5)) & (time.time() >= self.cut_time_list[i])  :
                do_action = '이제, ' + self.correct_act_ko[i] + ' 동작 해봐요!' # 한국어 버전
                draw.text((60, 30), f'{do_action.upper()}', font=font, fill=(255,255,255),stroke_width=3,stroke_fill=self.stroke_fill)
                img = np.array(img_pil)
        if time.time()>(self.cut_time_list[i+1]-2) :
                do_action = '이제 끝났어요! 안녕~~' # 한국어 버전
                draw.text((60, 30), f'{do_action.upper()}', font=font, fill=(255,255,255),stroke_width=3,stroke_fill=self.stroke_fill)
                img = np.array(img_pil)
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

                self.seq.append(d)

                self.mp_drawing.draw_landmarks(img, res, self.mp_hands.HAND_CONNECTIONS)

                if len(self.seq) < self.seq_length: # 설정한 시퀀스 킬이보다 클 경우만 이후 명령어 실행
                    continue

                input_data = np.expand_dims(np.array(self.seq[-self.seq_length:], dtype=np.float32), axis=0)

                y_pred = self.model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9: # 라벨에 대한 최대 예측 점수가 0.9가 넘어야 하는 조건
                    continue

                action = self.actions[i_pred]
                self.action_seq.append(action)

                if len(self.action_seq) < 3:
                    continue

                this_action = '?'
                if self.action_seq[-1] == self.action_seq[-2] == self.action_seq[-3]: # 연속된 3개의 action_seq가 들어와야 알맞는 동작으로 인식
                    this_action = action

                if this_action in self.correct_actions: # 출련된 action이 동요에 알맞는 action 인지 확인
                    idx = self.correct_actions.index(this_action)
                    font = ImageFont.truetype(self.fontpath, 35)
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), f'{self.correct_act_ko[idx].upper()}', font=font, fill=(255,255,255))
                    img = np.array(img_pil)
                    # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    c_time = time.time() 
                    # 동작 구분 시간을 구함으로써 중복 제거 방지
                    cut_off = [i for i in range(len(self.cut_time_list)-1) if (c_time>=self.cut_time_list[i])&(c_time<self.cut_time_list[i+1])]
                    if len(cut_off) != 0 :
                        if len(self.labels) == 0 :
                            self.labels.append([this_action,time.time(),cut_off])
                        else : 
                            if this_action != self.labels[-1][0]: # 전 동작과 같은 동작을 반복해서 인식하는 경우 하나의 동작 실행으로 받아들임
                                self.labels.append([this_action,time.time(),cut_off]) 
                            else :
                                if cut_off != self.labels[-1][-1]: # 반복하지만 동작 구분 시간이 다른 경우 다른 동작의 실행으로 받아들임
                                    self.labels.append([this_action,time.time(),cut_off]) 
        return  img, self.labels
