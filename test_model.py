# state = [1,0,...] # 42개 
import torch
import copy
import sys
import os
import numpy as np
from models import ResNetforDQN, AlphaZeroResNet, CFCNN, CFLinear, Classifier
from env import ConnectFour, Node, MCTS, Node_alphago, MCTS_alphago

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_class = {
    'easy':ResNetforDQN(action_size=49),  # Minimax DDQN with selfplay
    'normal':ResNetforDQN(action_size=49),  # AlphaGO with Minimax DDQN
    'hard':AlphaZeroResNet(3,5,128)  # AlphaZero
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

def get_encoded_state(state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)
    
    return encoded_state
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
# def load_model(model, device, filename='DQNmodel_CNN'):
#     model_path = ''
#     if filename.endswith(".pth") or filename.endswith(".pt"):
#         model_path = "model/"+filename
#     elif os.path.isfile("model/"+filename+".pth"):
#         model_path = "model/"+filename+".pth"
#     elif os.path.isfile("model/"+filename+".pt"):
#         model_path = "model/"+filename+".pt"
#     try:    
#         model.load_state_dict(torch.load(model_path, map_location=device))
#     except Exception as e:
#         print(f'모델 로드에서 예외가 발생했습니다: {e}')
            

# model 이름을 보고 어떤 type인지 확인 
def check_model_type(model_name):
    if 'Linear' in model_name:
        return 'Linear'
    elif 'CNN' in model_name:
        return 'CNN'
    else: 
        raise nameError


# def get_model_info(difficulty):
   
#     folder_path = 'model/'
#     folder_path = folder_path + difficulty + '/'
#     file_names = os.listdir(folder_path)

#     for file_name in file_names:
#         print(file_name)
#         if '.pt' in file_name:
#             model_name = file_name
#         elif '.json' in file_name:
#             config_name = file_name


#     return model_name, config_name

def load_model(difficulty):
    path = 'model/'+difficulty+'/'
    file_names = os.listdir(path)

    model = model_class[difficulty]

    if difficulty=='normal':
        for file_name in file_names:
            # print(file_name)
            if 'Value' in file_name:
                value_model_name = file_name
            elif '.pt' in file_name:
                model_name = file_name

        value_model = Classifier()

        try:    
            model.load_state_dict(
                torch.load(path+model_name, map_location=device)
            )
            value_model.load_state_dict(
                torch.load(path+value_model_name, map_location=device)
            )
        except Exception as e:
            print(f'모델 로드에서 예외가 발생했습니다: {e}')

        return (model, value_model)

        

        
    for file_name in file_names:
        # print(file_name)
        if '.pt' in file_name:
            model_name = file_name
        elif '.json' in file_name:
            config_name = file_name

    
    try:    
        model.load_state_dict(
            torch.load(path+model_name, map_location=device)
        )
    except Exception as e:
        print(f'모델 로드에서 예외가 발생했습니다: {e}')

    return model




def get_action(model, state, difficulty, vas):

    if isinstance(model, tuple):
        value_model = model[1].to(device)
        value_model.eval()
        model = model[0]

    model.to(device)
    model.eval()
    if isinstance(model, ResNetforDQN):
        if model.action_size==49:
            # 상황에 따라 다르게
            if difficulty=='normal':
                return get_alphago_action(model, value_model, state, vas)
            else:
                return get_minimax_action(model, state, vas)
            # return get_nash_action(model, state,vas)
        elif model.action_size==7:
            return get_DQN_action(model, state, vas)
    elif isinstance(model, AlphaZeroResNet):
        return get_alphazero_action(model, state, vas)
    elif isinstance(model, CFCNN) or isinstance(model, CFLinear):
        return get_DQN_action(model, state, vas)
    else: 
        print('error')
        exit()


def get_DQN_action(model, state, vas):
    q_values = model(state)
    # print(q_values)
    valid_q_values = q_values.squeeze()[torch.tensor(vas)]
    return vas[torch.argmax(valid_q_values)]

def get_minimax_action(model, state, valid_actions):

    q_values = model(
        torch.tensor(get_encoded_state(state.squeeze().cpu())).unsqueeze(0).to(device)
    )[0]
    # print(q_values.reshape(7,7))
    q_dict = {}
    # print(valid_actions)
    # print(distinct_actions)
    for a in valid_actions:
        q_dict[a] = []
        for b in valid_actions:
            idx = 7*a + b
            # print(a,b)
            # print(q_value[idx])
            # print(q_dict[a][1])
            
            q_dict[a].append((b, -q_values[idx]))

        q_dict[a] = torch.tensor(q_dict[a])
        op_action, value = q_dict[a][q_dict[a].argmax(dim=0)[1]]
        # op_action, value = softmax_policy(torch.tensor(q_dict[a]), temp=temp)
        # if torch.isnan(value):
        #     print(a,b)
        #     print(q_value.reshape(7,7))
        #     print(q_dict)
        q_dict[a] = (op_action, -1 * value)

    qs_my_turn = [[key, value[1]] for key, value in q_dict.items()]
    qs_my_turn = torch.tensor(qs_my_turn)
    action, value = qs_my_turn[qs_my_turn.argmax(dim=0)[1]]
        
    action = int(action)
    
    # action, value = softmax_policy(torch.tensor(qs_my_turn), temp=temp)
    # if torch.isnan(value):
    #         print(a,b)
    #         print(q_value.reshape(7,7))
    #         print(q_dict)

    return action


def get_nash_action(model,state, vas):
    q_values = model(state)

    pa, pb, v = get_nash_prob_and_value(q_values, vas)
    print(pa, pb, v)
    return np.random.choice(vas, p=pa)


def get_nash_prob_and_value(payoff_matrix, vas, iterations=100):
    if isinstance(payoff_matrix, torch.Tensor):    
        payoff_matrix = payoff_matrix.clone().detach().reshape(7,7)
    elif isinstance(payoff_matrix, np.ndarray):
        payoff_matrix = payoff_matrix.reshape(7,7)

    payoff_matrix = payoff_matrix[vas][:,vas]
    
    '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
    transpose_payoff = torch.transpose(payoff_matrix,0,1)
    row_cum_payoff = torch.zeros(len(payoff_matrix)).to(device)
    col_cum_payoff = torch.zeros(len(transpose_payoff)).to(device)

    col_count = np.zeros(len(transpose_payoff))
    row_count = np.zeros(len(payoff_matrix))
    active = 0
    for i in range(iterations):
        row_count[active] += 1 
        col_cum_payoff += payoff_matrix[active]
        active = torch.argmin(col_cum_payoff)
        col_count[active] += 1 
        row_cum_payoff += transpose_payoff[active]
        active = torch.argmax(row_cum_payoff)
        
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations  
    row_prob = row_count / iterations
    col_prob = col_count / iterations
    
    return row_prob, col_prob, value_of_game


def get_alphazero_action(model, state, vas):
    args = {
        'C': 1,
        'num_searches': 100,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }
    CF = ConnectFour()
    mcts = MCTS(CF, args, model)
    mcts_probs = mcts.search(state.squeeze().cpu().numpy())
    # action = np.random.choice(range(7),p=mcts_probs)
    action = np.argmax(mcts_probs)
    print(mcts_probs)
    
    return action

def get_alphago_action(model, value_model, state, vas):
    args = {
        'C': 1.5,
        'num_searches': 200,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    CF = ConnectFour()
    mcts = MCTS_alphago(CF, args, model, value_model)
    mcts_probs = mcts.search(state.squeeze().cpu().numpy())
    # action = np.random.choice(range(7),p=mcts_probs)
    action = np.argmax(mcts_probs)
    # print(mcts_probs)
    return action



def test_main(state, difficulty):
    # model type 확인

    # model_name, config_name = get_model_info(difficulty=difficulty)

    # print(model_name, config_name)
    # gpu 사용 여부 확인
    
    # device = torch.device("cpu")
    state = np.array(state)  # list to numpy array

    # 1p, 2p 확인
    player = check_player(state)

    # env가 없으므로 valid action이 뭔지 따로 확인
    valid_actions = get_valid_actions(state)
    # print("valid_actions:",valid_actions)
    # gradient 계산을 하지 않음
    with torch.no_grad():

        state = board_normalization(state, 'CNN', player).to(device)
        
        # alphago는 따로 적용 
        
        # 알맞은 model 할당
        model = load_model(difficulty)
        # print(model)
        # 가중치 load
        # 모델에 forward
        action = get_action(model, state, difficulty, valid_actions)
        # print("a:",action)

    # for debugging
    # print("model name:", model_name)
    # print("model type:", model_type)
    # print("player:", player)
    # print("Q values:", qvalues.tolist())
    # print("valid actions:", valid_actions)
    # print("maxQ:", torch.max(valid_q_values).item())
    # print("selected action:", valid_actions[torch.argmax(valid_q_values)])

    # 가장 높은 value를 가진 action return
    return action


if __name__ == "__main__":

    # 실행할 때 사용할 model의 이름을 적어줘야함
    # ex) python test_model.py DQNmodel_Linear
    # argvs = sys.argv
    # if len(argvs) == 1:
    #     model_name = 'DQNmodel_CNN'
    # else:
    #     model_name = argvs[1]

    # state 를 입력을 받음, 일단 test 용으로 2차원 배열 할당해놓음 
    # 1과 2로 이루어진 2차원 배열 

    # 현재 1을 놓아야하는 상태
    state = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,2,0,0,0],
        [0,0,0,1,1,0,0]
    ]


    # test_main의 인자는 state와 난이도로 이루어짐
    # 난이도는 'easy', 'normal', 'hard'
    print(test_main(state, 'hard'))