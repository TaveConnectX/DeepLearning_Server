# state = [1,0,...] # 42개 
import torch
import copy
import sys
import os
import numpy as np
from models import *


"""
사용법: python test_model.py ("model name")

model 폴더: 사용할 수 있는 model 모음
models.py: 모델의 구조를 알려주는 파일
test_model.py: 모델을 테스트할 수 있는 코드, 해당 state를 모델에 넣으면 그에 맞는 action(column)을 return 

유의점:
1. 현재 모델의 이름엔 "Linear", 또는 "CNN"이 포함되어 있어야함. 
일단 사용가능한 임시 모델들도 model이라는 폴더 만들어서 같이 push함
2. main 함수 인자 "model_name"은 생략가능하며 생략시 DQNmodel_CNN.pth 로 동작
3. 다른 state를 test하고 싶으면 main 함수 안의 2차원 list를 수정하면 됨 
4. 디버깅하고 싶으면 중간에 "for debugging" 아래의 print 문을 주석해제하면 편하게 정보 볼 수 있음
"""



# 이 models dict와 models.py 는 딥러닝 repository와 동일하게 유지되어야 함 
models = {
            1:CFLinear,
            2:CFCNN,
            3:HeuristicModel,
        }


# state가 정상적이지 않다면 error를 출력
class stateError(Exception):
    def __str__(self):
        return "impossible state"
    

# model의 이름이 적절하지 않으면 error를 출력
class nameError(Exception):
    def __str__(self):
        return "impossible model name"

# model의 type이 적절하지 않으면 error 출력 
class typeError(Exception):
    def __str__(self):
        return "impossible model type"
    
# model test를 위한 board_normalization() 함수 수정 버전
def board_normalization(state, model_type, player):
    # cnn을 사용하지 않는다면, 2차원 board를 1차원으로 바꿔줘야됨 
    
    if model_type == "Linear":
        arr = copy.deepcopy(state.flatten())
    elif model_type == "CNN": 
        arr = copy.deepcopy(state)



    """Replace all occurrences of 2 with -1 in a numpy array"""
    arr[arr == 2] = -1
    
    # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
    if player == 2: arr = -1 * arr

    arr = torch.from_numpy(arr).float()

    if model_type == "CNN":
        arr = arr.reshape(6,7).unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)

    return arr


# 보드판을 보고 지금이 누구의 턴인지 확인(1p, 2p)
def check_player(state):
    one = np.count_nonzero(state == 1)
    two = np.count_nonzero(state == 2)
    if one == two:
        return 1
    elif one == two+1:
        return 2
    else: raise stateError


# 보드판을 보고 가능한 action을 확인 (0~6)
def get_valid_actions(state):
    valid_actions = []
    for col in range(len(state[0])):
        if state[0][col]==0: 
            valid_actions.append(col)

    return valid_actions

# 모델 load. 매개변수만 load 하는게 overload가 적다고 하여 이 방법을 선택하였음 
def load_model(model, device, filename='DQNmodel_CNN'):
    print (device)
    if filename.endswith(".pth"):
        if device == torch.device('cpu') :
            model.load_state_dict(torch.load("model/"+filename, map_location=device))
        else :
            model.load_state_dict(torch.load("model/"+filename))
    elif filename.endswith(".pt"):
        if device == torch.device('cpu') :
            model.load_state_dict(torch.load("model/"+filename, map_location=device))
        else :
            model.load_state_dict(torch.load("model/"+filename))
    elif os.path.isfile("model/"+filename+".pth"):
        if device == torch.device('cpu') :
            model.load_state_dict(torch.load("model/"+filename+'.pth', map_location=device))
        else :
            model.load_state_dict(torch.load("model/"+filename+'.pth'))
    elif os.path.isfile("model/"+filename+".pt"):
        if device == torch.device('cpu') :
            model.load_state_dict(torch.load("model/"+filename+'.pt', map_location=device))
        else :
            model.load_state_dict(torch.load("model/"+filename+'.pt'))
    #except Exception as e:
    #    print(f'모델 로드에서 예외가 발생했습니다: {e}')
            

# model 이름을 보고 어떤 type인지 확인 
def check_model_type(model_name):
    if 'Linear' in model_name:
        return 'Linear'
    elif 'CNN' in model_name:
        return 'CNN'
    else: 
        raise nameError
    
def main():

    # 실행할 때 사용할 model의 이름을 적어줘야함
    # ex) python test_model.py DQNmodel_Linear
    argvs = sys.argv
    if len(argvs) == 1:
        model_name = 'DQNmodel_CNN'
    else:
        model_name = argvs[1]

    # model type 확인
    model_type = check_model_type(model_name)

    # gpu 사용 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # state 를 입력을 받음, 일단 test 용으로 2차원 배열 할당해놓음 
    # 1과 2로 이루어진 2차원 배열 

    # 현재 1을 놓아야하는 상태
    state = [
        [0,0,0,2,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,2,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,2,2,0,0,0],
        [0,0,1,1,1,2,0]
    ]

    state = np.array(state)  # list to numpy array

    # 1p, 2p 확인
    player = check_player(state)

    # env가 없으므로 valid action이 뭔지 따로 확인
    valid_actions = get_valid_actions(state)

    # gradient 계산을 하지 않음
    with torch.no_grad():

        state = board_normalization(state, model_type, player).to(device)
        

        # 모델 로드
        if model_type == "Linear": model_num = 1
        elif model_type == "CNN" : model_num = 2
        else: raise typeError

        # 알맞은 model 할당
        agent = models[model_num]().to(device)
        # 가중치 load
        load_model(agent, device, model_name)
        # 모델에 forward
        qvalues = agent(state)
        # 가능한 q value 모음 
        valid_q_values = qvalues.squeeze()[torch.tensor(valid_actions)]

    # for debugging
    # print("model name:", model_name)
    # print("model type:", model_type)
    # print("player:", player)
    # print("Q values:", qvalues.tolist())
    # print("valid actions:", valid_actions)
    # print("maxQ:", torch.max(valid_q_values).item())
    # print("selected action:", valid_actions[torch.argmax(valid_q_values)])


    # 가장 높은 value를 가진 action return
    return valid_actions[torch.argmax(valid_q_values)]

if __name__ == "__main__":
    print(main())