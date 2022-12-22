import numpy as np


class scoring_answer():
    def __init__(self, actions, correct_actions, labels, cut_time_list,level) :
        self.correct_actions = correct_actions # 정답 action
        self.labels = labels # 사용자 동작 결과
        self.cut_time_list = cut_time_list # 동작 구분 시간
        self.level = level
    
    def score(self) : # 레벨에 따른 다른 score 산출 방식
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
                self.labels.append([0,0,0])
        labels = np.array(self.labels,dtype=object)
        print(labels)
        labels = labels[:,:-1] # 동작 구분 시간 결과값 떼어내기

        if self.level == 'easy' :
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
                            if self.correct_actions[i] in labels[:,0] :
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
        elif self.level == 'hard' :
            # 'easy' mode와 실행 원리는 같음 (동작 구분 시작의 구간이 더욱 짧아진 것만 다름)
            for i in range(num_correct) :
                if self.correct_actions[i] == labels[i,0] :
                    if (float(labels[i][1])>=self.cut_time_list[i]) & (float(labels[i][1])<=self.cut_time_list[i+1]) :
                        score += 1
                    else : 
                        if i == len(labels) :
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