from neural_network import DQN
from player import Player
import settings
import torch
from utils import TablutUtils

utils = TablutUtils()

model = DQN(settings.NUM_OBSERVATIONS, settings.NUM_MOVES).to(device=settings.device)

model.load_state_dict(torch.load("./models/white5000.pt", map_location=settings.device))

player = Player("WHITE123", True)

player.connect()
player.declare_name()

over = False

while not over:
    current_state = player.current_state()

    board = current_state['board']

    current_turn = current_state['turn']

    if current_turn == "WHITEWIN" or current_turn == "BLACKWIN" or current_turn == "DRAW":
        break

    if current_turn != "WHITE":
        continue
    converted_board_matrix = utils.convert_current_board(board)
    converted_board = converted_board_matrix.flatten()
    state_tensor = torch.tensor(converted_board, dtype=torch.float32, device=settings.device).unsqueeze(0)
    print(state_tensor)
    with torch.no_grad():
        state = state_tensor
        q_values = model(state)
        action_index = q_values.argmax().item()
        action = torch.tensor([[q_values.argmax()]], device=settings.device, dtype=torch.long)

    cell_from = chr(ord('a') + ((action_index // 81) % 9)) + str((action_index // 81) // 9 + 1) 
    cell_to = chr(ord('a') + ((action_index % 81) % 9)) + str((action_index % 81) // 9 + 1)
    next_move = {'from': cell_from, 'to': cell_to}
    player.make_move(next_move)
player.close()
