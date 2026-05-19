"""
Kronos — Connect-4 Agent
========================
Hybrid agent: Minimax with Alpha-Beta pruning + online Q-Learning
that adapts heuristic weights during play.

Configurable knobs (for experimental analysis):
  depth            : minimax search depth (resource proxy)
  use_q_learning   : enable/disable online weight update module
  learning_rate    : alpha for TD(0) weight updates
  exploration_rate : epsilon for epsilon-greedy exploration
"""

from __future__ import annotations

import numpy as np

from connect4.policy import Policy

ROWS = 6
COLS = 7
RED    = -1
YELLOW =  1


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

def _windows(board: np.ndarray):
    """Yield all length-4 windows (horizontal, vertical, diagonal)."""
    for r in range(ROWS):
        for c in range(COLS - 3):
            yield board[r, c:c+4]
    for c in range(COLS):
        for r in range(ROWS - 3):
            yield board[r:r+4, c]
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            yield board[[r, r+1, r+2, r+3], [c, c+1, c+2, c+3]]
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            yield board[[r, r-1, r-2, r-3], [c, c+1, c+2, c+3]]


def _window_score(window: np.ndarray, player: int) -> float:
    opp     = -player
    p_count = np.count_nonzero(window == player)
    o_count = np.count_nonzero(window == opp)
    empty   = np.count_nonzero(window == 0)

    if o_count > 0 and p_count > 0:
        return 0.0
    if p_count == 4:
        return 1000.0
    if p_count == 3 and empty == 1:
        return 5.0
    if p_count == 2 and empty == 2:
        return 2.0
    if o_count == 3 and empty == 1:
        return -8.0
    return 0.0


def extract_features(board: np.ndarray, player: int) -> np.ndarray:
    opp    = -player
    centre = board[:, COLS // 2]
    f0     = float(np.count_nonzero(centre == player))
    f1 = f2 = f3 = f4 = 0.0

    for w in _windows(board):
        p = np.count_nonzero(w == player)
        o = np.count_nonzero(w == opp)
        e = np.count_nonzero(w == 0)

        if o == 0:
            if p == 2 and e == 2: f2 += 1
            if p == 3 and e == 1: f3 += 1
        if p == 0 and o == 3 and e == 1:
            f4 += 1

        f1 += _window_score(w, player)

    return np.array([f0, f1, f2, f3, f4], dtype=np.float64)


def evaluate(board: np.ndarray, player: int, weights: np.ndarray) -> float:
    return float(weights @ extract_features(board, player))


# ──────────────────────────────────────────────────────────────
# Game helpers
# ──────────────────────────────────────────────────────────────

def get_free_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def drop_piece(board: np.ndarray, col: int, player: int) -> np.ndarray:
    b = board.copy()
    for r in reversed(range(ROWS)):
        if b[r, col] == 0:
            b[r, col] = player
            return b
    raise ValueError("Column full")


def check_win(board: np.ndarray, player: int) -> bool:
    for w in _windows(board):
        if np.all(w == player):
            return True
    return False


def is_terminal(board: np.ndarray) -> bool:
    return (check_win(board, RED) or check_win(board, YELLOW)
            or len(get_free_cols(board)) == 0)


# ──────────────────────────────────────────────────────────────
# Minimax with alpha-beta pruning
# ──────────────────────────────────────────────────────────────

def minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
    player: int,
    weights: np.ndarray,
) -> tuple[float, int | None]:
    free = get_free_cols(board)

    if is_terminal(board):
        if check_win(board, player):     return (1e9, None)
        if check_win(board, -player):    return (-1e9, None)
        return (0.0, None)

    if depth == 0:
        return (evaluate(board, player, weights), None)

    free_sorted = sorted(free, key=lambda c: -abs(c - COLS // 2) + COLS)

    if maximising:
        best_score, best_col = -np.inf, free_sorted[0]
        for col in free_sorted:
            child = drop_piece(board, col, player)
            score, _ = minimax(child, depth-1, alpha, beta, False, player, weights)
            if score > best_score:
                best_score, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return (best_score, best_col)
    else:
        best_score, best_col = np.inf, free_sorted[0]
        for col in free_sorted:
            child = drop_piece(board, col, -player)
            score, _ = minimax(child, depth-1, alpha, beta, True, player, weights)
            if score < best_score:
                best_score, best_col = score, col
            beta = min(beta, score)
            if alpha >= beta:
                break
        return (best_score, best_col)


# ──────────────────────────────────────────────────────────────
# Kronos Policy
# ──────────────────────────────────────────────────────────────

class Kronos(Policy):
    """
    Kronos Connect-4 agent.

    Parameters
    ----------
    depth : int
        Minimax search depth. Higher = stronger but slower.
    use_q_learning : bool
        Enable online TD(0) weight updates after each move.
    learning_rate : float
        Alpha for Q-Learning weight updates.
    exploration_rate : float
        Epsilon for epsilon-greedy exploration.
    seed : int | None
        RNG seed for reproducibility.
    """

    _DEFAULT_WEIGHTS = np.array([3.0, 1.0, 2.0, 5.0, -4.0], dtype=np.float64)

    def __init__(
        self,
        depth: int = 2,
        use_q_learning: bool = True,
        learning_rate: float = 0.05,
        exploration_rate: float = 0.0,
        seed: int | None = 42,
    ):
        self.depth            = depth
        self.use_q_learning   = use_q_learning
        self.learning_rate    = learning_rate
        self.exploration_rate = exploration_rate
        self.seed             = seed

        self._weights:    np.ndarray = self._DEFAULT_WEIGHTS.copy()
        self._rng:        np.random.Generator = np.random.default_rng(self.seed)
        self._player:     int | None = None
        self._prev_board: np.ndarray | None = None
        self._prev_eval:  float = 0.0


    def mount(self, timeout: float | None = None) -> None:
        self._weights    = self._DEFAULT_WEIGHTS.copy()
        self._rng        = np.random.default_rng(self.seed)
        self._player     = None
        self._prev_board = None
        self._prev_eval  = 0.0

    def _detect_player(self, board: np.ndarray) -> int:
        red_count    = np.count_nonzero(board == RED)
        yellow_count = np.count_nonzero(board == YELLOW)
        return RED if red_count == yellow_count else YELLOW

    def _immediate_win_or_block(self, board: np.ndarray, player: int) -> int | None:
        for col in get_free_cols(board):
            if check_win(drop_piece(board, col, player), player):
                return col
        for col in get_free_cols(board):
            if check_win(drop_piece(board, col, -player), -player):
                return col
        return None

    def _q_update(self, board: np.ndarray) -> None:
        if self._prev_board is None:
            return
        current_eval = evaluate(board, self._player, self._weights)
        td_error     = current_eval - self._prev_eval
        grad         = extract_features(self._prev_board, self._player)
        self._weights += self.learning_rate * td_error * grad

    def act(self, s: np.ndarray) -> int:
        board = s.copy()

        if self._player is None:
            self._player = self._detect_player(board)

        if self.use_q_learning and self._prev_board is not None:
            self._q_update(board)

        forced = self._immediate_win_or_block(board, self._player)
        if forced is not None:
            self._prev_board = board
            self._prev_eval  = evaluate(board, self._player, self._weights)
            return forced

        if self._rng.random() < self.exploration_rate:
            col = int(self._rng.choice(get_free_cols(board)))
            self._prev_board = board
            self._prev_eval  = evaluate(board, self._player, self._weights)
            return col

        _, best_col = minimax(
            board, self.depth, -np.inf, np.inf,
            True, self._player, self._weights,
        )

        if best_col is None:
            best_col = get_free_cols(board)[0]

        self._prev_board = board
        self._prev_eval  = evaluate(board, self._player, self._weights)
        return int(best_col)
