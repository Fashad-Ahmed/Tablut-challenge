import random

import torch
import torch.optim as optim
from neural_network import DQN
from player import Player
from replay_memory import ReplayMemory, Transition
import settings
from utils import TablutUtils


class Agent:
    def __init__(self, is_white:bool = True, name: str = "Agent", load_prev_path = None):
        self.is_white = is_white
        self.player = Player(name, is_white)
        self.memory = ReplayMemory(settings.MEMORY_SIZE)
        self.policy_dqn = DQN(settings.NUM_OBSERVATIONS, settings.NUM_MOVES).to(settings.device)
        
        if load_prev_path != None:
            self.policy_dqn.load_state_dict(torch.load(load_prev_path, map_location=settings.device))
        
        self.target_dqn = DQN(settings.NUM_OBSERVATIONS, settings.NUM_MOVES).to(settings.device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.optimizer = optim.AdamW(self.policy_dqn.parameters(), lr=settings.LR)


        self.utils = TablutUtils()

        self.turn = "WHITE" if is_white else "BLACK"
        self.steps_agent = 0
        self.old_state = None
        self.action = None
        self.state_tensor = None
        self.old_action_index = None
        self.current_action_index = None
        self.illegal_move = False

    def connect(self):
        self.player.connect()
        self.player.declare_name()
    
    def close(self):
        self.player.close()

    def read_current_state(self):
        return self.player.current_state()

    def calculate_move(self) -> str:
        eps = self.utils.compute_epsilon(self.steps_agent)
        self.steps_agent += 1
        legal_moves = self.utils.generate_legale_moves(self.is_white, self.converted_board_matrix)
        self.illegal_move = False
        if random.random() > eps:
            with torch.no_grad():
                state = self.state_tensor
                q_values = self.policy_dqn(state)
                action_index = q_values.argmax().item()
                self.current_action_index = action_index
                self.illegal_move = action_index not in legal_moves
                self.action = torch.tensor([[q_values.argmax()]], device=settings.device, dtype=torch.long)
                
        else:
            action_index = random.choice(legal_moves) 
            self.current_action_index = action_index
            self.action = torch.tensor([[action_index]], device=settings.device, dtype=torch.long)
        
        cell_from = chr(ord('a') + ((action_index // 81) % 9)) + str((action_index // 81) // 9 + 1) 
        cell_to = chr(ord('a') + ((action_index % 81) % 9)) + str((action_index % 81) // 9 + 1)
        return {'from': cell_from, 'to': cell_to}

    def store_state(self):
        self.old_state = self.state_tensor

    def store_action(self):
        self.old_action_index = self.current_action_index

    def read_state(self):
        self.current_state = self.read_current_state()
        self.board = self.current_state['board']
        self.current_turn = self.current_state['turn']
        self.converted_board_matrix = self.utils.convert_current_board(self.board)
        self.converted_board = self.converted_board_matrix.flatten()
        self.state_tensor = torch.tensor(self.converted_board, dtype=torch.float32, device=settings.device).unsqueeze(0)#self.utils.state_to_tensor(self.converted_board)

    def make_move(self):
        if self.current_turn == self.turn and not self.win() and not self.draw():
            self.store_state()
            next_move = self.calculate_move()
            self.player.make_move(next_move)

    def win(self):
        if self.is_white:
            if self.current_turn == "WHITEWIN":
                return True
        else:
            if self.current_turn == "BLACKWIN":
                return True
        return False
    
    def lost(self):
        if self.is_white:
            if self.current_turn == "BLACKWIN":
                return True
        else:
            if self.current_turn == "WHITEWIN":
                return True
        return False
    
    def draw(self):
        if self.current_turn == "DRAW":
                return True
        return False
    
    def reward(self) -> float:
        if self.win():
            return 1
        elif self.lost():
            return -3.0
        elif self.illegal_move or self.old_action_index == self.current_action_index:
            return -10.0
        return -0.001

    def optimize_model(self):
        if len(self.memory) < settings.BATCH_SIZE:
            return
        transitions = self.memory.sample(settings.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=settings.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        try:
            action = self.policy_dqn(state_batch)
            state_action_values = action.gather(1, action_batch)
        except:
            print(state_batch)
            print(action_batch)
            raise MemoryError

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(settings.BATCH_SIZE, device=settings.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * settings.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_dqn.parameters(), 100)
        self.optimizer.step()

    def train(self):
        reward = torch.tensor([self.reward()], dtype=torch.float32, device=settings.device)
        
        if self.old_state != None and self.state_tensor != None:
            self.memory.push(self.old_state, self.action, self.state_tensor, reward)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_dqn.state_dict()
            policy_net_state_dict = self.policy_dqn.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*settings.TAU + target_net_state_dict[key]*(1-settings.TAU)
            self.target_dqn.load_state_dict(target_net_state_dict)

    def save_model(self, name = ""):
        torch.save(self.policy_dqn.state_dict(), f"./models/white{name}.pt" if self.is_white else f"./models/black{name}.pt")
        