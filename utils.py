import math
import random
import torch
import settings
import numpy as np


class TablutUtils:
    def convert_current_board(self, board) -> np.ndarray:
        board_converted = np.array([[np.around(settings.EMPTY_TENSOR_VALUE, 2)]*9]*9)
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == "WHITE":
                    board_converted[i,j] = np.around(settings.WHITE_TENSOR_VALUE, 2)
                elif board[i][j] == "KING":
                    board_converted[i,j] = np.around(settings.KING_TENSOR_VALUE, 2)
                elif board[i][j] == "BLACK":
                    board_converted[i,j] = np.around(settings.BLACK_TENSOR_VALUE, 2)
        return board_converted
    
    def state_to_tensor(self, board) -> torch.Tensor:
        """Try to convert a board representation from the server into a flat
        tensor of length settings.NUM_OBSERVATIONS. The function is permissive:
        - If `board` is a list/tuple, it will be flattened and truncated/padded.
        - If `board` is a dict with key 'board' it will use that.
        - If `board` is a string of digits it will be parsed.
        If conversion fails, returns a zero tensor.
        """
        n = settings.NUM_OBSERVATIONS
        try:
            if isinstance(board, dict) and 'board' in board:
                data = board['board']
            else:
                data = board

            if isinstance(data, (np.ndarray, list, tuple)):
                flat = []
                for v in data:
                    if isinstance(v, (np.ndarray,list, tuple)):
                        flat.extend(v)
                    else:
                        flat.append(v)
                arr = [float(x) for x in flat[:n]]
                if len(arr) < n:
                    arr += [0.0] * (n - len(arr))
                return torch.tensor(arr, dtype=torch.float32, device=settings.device)

            if isinstance(data, str):
                print(2)
                # try to pick digits
                digits = [float(ch) for ch in data if ch.isdigit()]
                arr = digits[:n]
                if len(arr) < n:
                    arr += [0.0] * (n - len(arr))
                return torch.tensor(arr, dtype=torch.float32, device=settings.device)

            # fallback: zero vector
        except Exception:
            pass
        return torch.zeros(n, dtype=torch.float32, device=settings.device)
    
    def compute_epsilon(self, steps_done: int) -> float:
        return settings.EPS_END + (settings.EPS_START - settings.EPS_END) * math.exp(-1.0 * steps_done / settings.EPS_DECAY)
    
    def generate_legale_moves(self, is_white: bool, board: np.ndarray):
        legal_moves = []
        pawns = []

        if is_white:
            pawns = list(map(tuple, np.argwhere(np.isin(board, [np.around(settings.WHITE_TENSOR_VALUE, 2), np.around(settings.KING_TENSOR_VALUE, 2)]))))
        else:
            pawns = list(map(tuple, np.argwhere(np.isin(board, [np.around(settings.BLACK_TENSOR_VALUE, 2)]))))

        empty = list(map(tuple, np.argwhere(np.isin(board, [np.around(settings.EMPTY_TENSOR_VALUE, 2)]))))

        for i in range(len(pawns)):
            cell_pawn = pawns[i][0] * 9 + pawns[i][1]
            for j in range(len(empty)):
                if pawns[i][0] != empty[j][0] and pawns[i][1] != empty[j][1]:
                    continue

                if pawns[i][0] == empty[j][0]:
                    def check_move(from_cell, to_cell):
                        tmp = from_cell
                        while tmp < to_cell:
                            if board[pawns[i][0], tmp]:
                                return True
                            tmp+=1
                        return False
                    
                    is_legal = (pawns[i][1] < empty[j][1] and check_move(pawns[i][1], empty[j][1])) or (pawns[i][1] > empty[j][1] and check_move(empty[j][1], pawns[i][1]))

                    if not is_legal:
                        continue
                elif pawns[i][1] == empty[j][1]:
                    def check_move(from_cell, to_cell):
                        tmp = from_cell
                        while tmp < to_cell:
                            if board[tmp, pawns[i][1]]:
                                return True
                            tmp+=1
                        return False
                    
                    is_legal = (pawns[i][0] < empty[j][0] and check_move(pawns[i][0], empty[j][0])) or (pawns[i][0] > empty[j][0] and check_move(empty[j][0], pawns[i][0]))

                    if not is_legal:
                        continue
                cell_empty = empty[j][0] * 9 + empty[j][1]
                move = cell_pawn * 81 + cell_empty
                legal_moves.append(move)
        random.shuffle(legal_moves)
        return legal_moves