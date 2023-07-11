import torch
import numpy as np
import math


def get_encoded_state(state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)
    
    if len(state.shape) == 3:
        encoded_state = np.swapaxes(encoded_state, 0, 1)
    
    return encoded_state

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            # child와 parent는 적이므로 1에서 빼주기로 한다 
            q_value = -(child.value_sum / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                # 내가 두는 건 항상 1, child 는 -1이면 뭔가 이상한데,,,
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                # game, args, state, parent=None, action_taken=None, prior=0, visit_count=0
                child = Node(
                    game=self.game, 
                    args=self.args, 
                    state=child_state, 
                    parent=self, 
                    action_taken=action, 
                    prior=prob
                )
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class Node_alphago:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            # child와 parent는 적이므로 1에서 빼주기로 한다 
            q_value = -(child.value_sum / child.visit_count)
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                # 내가 두는 건 항상 1, child 는 -1이면 뭔가 이상한데,,,
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                # game, args, state, parent=None, action_taken=None, prior=0, visit_count=0
                child = Node_alphago(
                    game=self.game, 
                    args=self.args, 
                    state=child_state, 
                    parent=self, 
                    action_taken=action, 
                    prior=prob
                )
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state), 
                        device=self.model.device
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        # action prob은 방문 횟수에 비례하도록 정한다 
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    


class MCTS_alphago:
    def __init__(self, game, args, model, value_model):
        self.game = game
        self.args = args
        self.model = model
        self.value_model = value_model

    def board_normalization(self,state):
        return torch.tensor(state, device=self.model.device).float().unsqueeze(0).unsqueeze(0)


    
    def get_nash_prob_and_value(self,payoff_matrix, vas, iterations=50):
        if isinstance(payoff_matrix, torch.Tensor):    
            payoff_matrix = payoff_matrix.clone().detach().reshape(7,7)
        elif isinstance(payoff_matrix, np.ndarray):
            payoff_matrix = payoff_matrix.reshape(7,7)
        vas = np.where(np.array(vas) == 1)[0]
        payoff_matrix = payoff_matrix[vas][:,vas]
        # print("vas:",vas)
        '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
        transpose_payoff = torch.transpose(payoff_matrix,0,1)
        row_cum_payoff = torch.zeros(len(payoff_matrix)).to(self.model.device)
        col_cum_payoff = torch.zeros(len(transpose_payoff)).to(self.model.device)

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
    

    def softmax(self, lst, temperature=1.0):
        # Scale the input values by the temperature
        scaled_lst = [x / temperature for x in lst]
        
        # Compute the sum of exponential values for each element
        exp_sum = sum(math.exp(x) for x in scaled_lst)
        
        # Apply softmax function for each element
        softmax_lst = [math.exp(x) / exp_sum for x in scaled_lst]
        
        return softmax_lst
    
    def get_minimax_prob_and_value(self, q_value, vas):
        # q_value = q_value.clone().detach().reshape(7,7)
        q_value = q_value.squeeze()
        vas = np.where(np.array(vas) == 1)[0]
        # q_value = q_value[vas][:,vas]
        q_dict = {}
        for a in vas:
            q_dict[a] = []
            for b in vas:
                idx = 7*a + b

                q_dict[a].append((b, -q_value[idx]))
            
            maxidx = torch.tensor(q_dict[a]).argmax(dim=0)[1]

            op_action, value = q_dict[a][maxidx]
            q_dict[a] = (op_action, -1*value)

        qs_my_turn = [value[1] for key, value in q_dict.items()]
        
        policy = self.softmax(qs_my_turn, temperature=0.05)
        value = max(qs_my_turn)

        return policy, value
        

    @torch.no_grad()
    def search(self, state):
        root = Node_alphago(self.game, self.args, state, visit_count=1)
        
        # policy 만드는 부분을 바꿔야됨 
        q_values = self.model(
            torch.tensor(get_encoded_state(state)).unsqueeze(0).to(self.model.device)
        )
        valid_moves = self.game.get_valid_moves(state)
        # print(q_values)
        # print(valid_moves)
        # pa, pb, v = self.get_nash_prob_and_value(q_values, valid_moves)
        pa, v = self.get_minimax_prob_and_value(q_values, valid_moves)
        policy = np.zeros_like(valid_moves, dtype=float)
        policy[np.array(valid_moves) == 1] = pa
        print(policy, v)
        # print(np.array(valid_moves) == 1,policy,pa,pb, v)
        # policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        # policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
        #     * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        policy *= valid_moves
        policy /= policy.sum()

        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                q_values = self.model(
                    torch.tensor(get_encoded_state(node.state)).unsqueeze(0).to(self.model.device)
                )
                valid_moves = self.game.get_valid_moves(node.state)
                # print(node.state, valid_moves)
                # print(q_values)
                # print(valid_moves)
                # pa, pb, value = self.get_nash_prob_and_value(q_values, valid_moves)
                pa, value = self.get_minimax_prob_and_value(q_values, valid_moves)
                policy = np.zeros_like(valid_moves, dtype=float)
                policy[np.array(valid_moves) == 1] = pa
            #     policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            # * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
                
                policy *= valid_moves
                policy /= policy.sum()
                # print(policy,pb, value)
                # policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                # valid_moves = self.game.get_valid_moves(node.state)
                # policy *= valid_moves
                # policy /= np.sum(policy)
                
                value = value.item()
                node.expand(policy)
                # 여기서 rollout policy로 다 둬보기
                # value_r = self.get_rollout_value(node.state)
                # rollout policy는 컴퓨팅 파워가 많이 필요하므로 nash value로 대체 
                value_r = value
                # value network에 넣어보기 
                # value_from_net = self.get_value_from_net(node.state)
                # value_net 이 완성되기 전까진 nash value로 대체
                value_from_net = self.get_value_from_net(node.state)
                
                
                # 둘을 평균낸 것을 value로 쓴다
                value = (1-0.2) * value_r + 0.2 * value_from_net
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        # action prob은 방문 횟수에 비례하도록 정한다 
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
    def get_rollout_value(self, state):
        # 끝날 때까지 둬보기
        # 시간을 매우 많이 잡아먹으므로 Q-value 로 대체
        pass
    
    def get_value_from_net(self, state):
        v_idx = torch.argmax(self.value_model(torch.FloatTensor(state).flatten().to(self.model.device)))
        if v_idx==0: value_from_net = 1
        elif v_idx==1: value_from_net = 0
        elif v_idx==2: value_from_net = -1
        else: exit()

        return value_from_net
