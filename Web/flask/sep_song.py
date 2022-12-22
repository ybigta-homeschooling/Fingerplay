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
# 동요별 cut_time
cut_time_3 = [3,14,18,35]
cut_time_1 = [0,8,15,25]
cut_time_4 = [4,23,31,38]
cut_time_2 = [0,6,13,20,29] 
# 동요별 글자 색깔
stroke_fill_3 = (81,164,75)
stroke_fill_1 = (85,82,246)
stroke_fill_4 = (241,175,76)
stroke_fill_2 = (69,91,125)

def song_by_song(song_num,level='easy'):
    # 노래에 따른 변수 생성
    song_num_str=str(song_num)
    correct_actions = globals()['correct_actions_' + song_num_str]
    correct_act_ko = globals()['correct_act_ko_' + song_num_str]
    cut_time_original = globals()['cut_time_' + song_num_str]
    plus_list = [6] * len(cut_time_original) # 카메라 delay 시간
    cut_time = [a+b for a,b in zip(cut_time_original,plus_list)] # action 실행 구간
    stroke_fill = globals()['stroke_fill_' + song_num_str]

    return correct_actions, correct_act_ko, cut_time, stroke_fill
