import sys
sys.path.insert(0, '.')

from policy import Kronos, get_free_cols, apply_move, check_winner, is_terminal, ROWS, COLS
import numpy as np

agente = Kronos()
agente.mount()  # carga el .pkl, instantáneo

def random_pol(board):
    return int(np.random.choice(get_free_cols(board)))

# 200 partidas como Rojo
wins, draws, losses = 0, 0, 0
for _ in range(200):
    board = np.zeros((ROWS, COLS), dtype=int)
    player = -1
    while not is_terminal(board):
        col = agente.act(board) if player == -1 else random_pol(board)
        board = apply_move(board, col, player)
        player = -player
    w = check_winner(board)
    if w == -1: wins += 1
    elif w == 0: draws += 1
    else: losses += 1

print(f"Como ROJO   — Victorias: {wins}/200 | Empates: {draws} | Derrotas: {losses}")

# 200 partidas como Amarillo
wins, draws, losses = 0, 0, 0
for _ in range(200):
    board = np.zeros((ROWS, COLS), dtype=int)
    player = -1
    while not is_terminal(board):
        col = random_pol(board) if player == -1 else agente.act(board)
        board = apply_move(board, col, player)
        player = -player
    w = check_winner(board)
    if w == 1: wins += 1
    elif w == 0: draws += 1
    else: losses += 1

print(f"Como AMARILLO — Victorias: {wins}/200 | Empates: {draws} | Derrotas: {losses}")