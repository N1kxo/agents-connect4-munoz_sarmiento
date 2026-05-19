"""
Katsura – MCTS Agent for Connect-4
===================================
Monte-Carlo Tree Search with UCB1 exploration for Connect-4.

Based on concepts from:
  - Slide 13 (Online Policy Improvement / MCTS)
  - Slide 12 (Competitive MDPs, UCB-based exploration in Bernoulli games)

How it works
------------
On every call to act(), Katsura runs MCTS simulations within a time budget.
Each episode has four phases:

  1. Selection   – walk the tree using UCB1 until a node that still has
                   unvisited children (a "not fully expanded" node) or a
                   terminal node is reached.
  2. Expansion   – add one new child node for an unvisited action.
  3. Rollout     – play out the game randomly from the new node.
  4. Backprop    – propagate the result (+1 win / 0 draw / -1 loss) back
                   up the path, flipping sign at each level (zero-sum).

After the budget is exhausted, Katsura picks the action with the highest
visit count N (most robust choice).

"Tornillos" (configuration knobs for the analysis notebook):
  n_simulations : int   – MCTS budget per move (more = stronger, slower)
  c_exploration : float – UCB1 exploration constant (√2 by default)
  rollout_policy: str   – 'random' | 'heuristic'
  time_limit    : float – hard time cap per move in seconds (None = no cap)
"""

import math
import random
import time
import numpy as np
from abc import ABC
from typing import Optional, List

try:
    from typing import override
except ImportError:
    def override(f): return f

try:
    from connect4.policy import Policy
except ImportError:
    class Policy(ABC):
        def mount(self, timeout=None): pass
        def act(self, s): pass


# ============================================================================
# Board constants
# ============================================================================

ROWS = 6
COLS = 7
EMPTY = 0

# Precompute all winning 4-in-a-row index sets for fast winner detection
_WIN_INDICES = []
for r in range(ROWS):
    for c in range(COLS):
        if c + 3 < COLS:
            _WIN_INDICES.append([(r, c+i) for i in range(4)])
        if r + 3 < ROWS:
            _WIN_INDICES.append([(r+i, c) for i in range(4)])
        if r + 3 < ROWS and c + 3 < COLS:
            _WIN_INDICES.append([(r+i, c+i) for i in range(4)])
        if r + 3 < ROWS and c - 3 >= 0:
            _WIN_INDICES.append([(r+i, c-i) for i in range(4)])

# Convert to numpy arrays for fast vectorized checks
_WIN_ROWS = np.array([[idx[0] for idx in w] for w in _WIN_INDICES])
_WIN_COLS = np.array([[idx[1] for idx in w] for w in _WIN_INDICES])


# ============================================================================
# Fast board operations
# ============================================================================

def _drop(board: np.ndarray, col: int, player: int) -> Optional[np.ndarray]:
    if board[0, col] != EMPTY:
        return None
    new = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if new[r, col] == EMPTY:
            new[r, col] = player
            break
    return new


def _drop_inplace(board: np.ndarray, col: int, player: int) -> int:
    """Drop piece inplace, return the row where it landed (-1 if invalid)."""
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == EMPTY:
            board[r, col] = player
            return r
    return -1


def _free_cols(board: np.ndarray) -> List[int]:
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def _winner_fast(board: np.ndarray) -> int:
    """Vectorized winner check using precomputed indices."""
    vals = board[_WIN_ROWS, _WIN_COLS]  # shape (N_WINS, 4)
    row_sums = vals.sum(axis=1)
    if np.any(row_sums == 4):
        return 1
    if np.any(row_sums == -4):
        return -1
    return 0


def _winner_after_drop(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Fast check: did placing player at (row,col) create a win?"""
    # Only check lines that pass through (row, col)
    p = player
    # Horizontal
    r = row
    count = 0
    for c in range(max(0, col-3), min(COLS, col+4)):
        count = count + 1 if board[r, c] == p else 0
        if count >= 4:
            return True
    # Vertical
    count = 0
    for rr in range(max(0, row-3), min(ROWS, row+4)):
        count = count + 1 if board[rr, col] == p else 0
        if count >= 4:
            return True
    # Diag down-right
    count = 0
    for i in range(-3, 4):
        rr, cc = row+i, col+i
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            count = count + 1 if board[rr, cc] == p else 0
            if count >= 4:
                return True
    # Diag down-left
    count = 0
    for i in range(-3, 4):
        rr, cc = row+i, col-i
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            count = count + 1 if board[rr, cc] == p else 0
            if count >= 4:
                return True
    return False


def _heuristic_move(board: np.ndarray, player: int) -> Optional[int]:
    cols = _free_cols(board)
    for c in cols:
        for r in range(ROWS - 1, -1, -1):
            if board[r, c] == EMPTY:
                board[r, c] = player
                win = _winner_after_drop(board, r, c, player)
                board[r, c] = EMPTY
                if win:
                    return c
                break
    opp = -player
    for c in cols:
        for r in range(ROWS - 1, -1, -1):
            if board[r, c] == EMPTY:
                board[r, c] = opp
                win = _winner_after_drop(board, r, c, opp)
                board[r, c] = EMPTY
                if win:
                    return c
                break
    return None


# ============================================================================
# Fast rollout using inplace ops
# ============================================================================

def _rollout_fast(board: np.ndarray, player: int, policy: str) -> int:
    b = board.copy()
    current = player
    while True:
        free = [c for c in range(COLS) if b[0, c] == EMPTY]
        if not free:
            return 0
        if policy == "heuristic":
            col = _heuristic_move(b, current)
            if col is None:
                col = random.choice(free)
        else:
            col = random.choice(free)
        row = _drop_inplace(b, col, current)
        if _winner_after_drop(b, row, col, current):
            return current
        current = -current


# ============================================================================
# MCTS Node
# ============================================================================

class _Node:
    __slots__ = ("board", "player", "parent", "action",
                 "children", "untried", "wins", "visits")

    def __init__(self, board: np.ndarray, player: int,
                 parent: Optional["_Node"] = None, action: Optional[int] = None):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children: List["_Node"] = []
        self.untried: List[int] = _free_cols(board)
        random.shuffle(self.untried)
        self.wins: float = 0.0
        self.visits: int = 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return _winner_fast(self.board) != 0 or len(_free_cols(self.board)) == 0

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.wins / self.visits) + c * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def best_child(self, c: float) -> "_Node":
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self) -> "_Node":
        col = self.untried.pop()
        new_board = _drop(self.board, col, self.player)
        child = _Node(new_board, -self.player, parent=self, action=col)
        self.children.append(child)
        return child

    def backpropagate(self, result: float, root_player: int) -> None:
        node = self
        while node is not None:
            node.visits += 1
            if node.player != root_player:
                node.wins += result
            else:
                node.wins -= result
            node = node.parent


# ============================================================================
# MCTS runner
# ============================================================================

def mcts(board: np.ndarray, root_player: int,
         n_simulations: int, c_exploration: float,
         rollout_policy: str, time_limit: Optional[float] = None) -> int:
    root = _Node(board, root_player)
    deadline = time.time() + time_limit if time_limit else None

    for i in range(n_simulations):
        if deadline and time.time() >= deadline:
            break

        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(c_exploration)

        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        winner = _rollout_fast(node.board, node.player, rollout_policy)

        result = 1.0 if winner == root_player else (-1.0 if winner != 0 else 0.0)
        node.backpropagate(result, root_player)

    best = max(root.children, key=lambda n: n.visits)
    return best.action


# ============================================================================
# Policy class
# ============================================================================

class Katsura(Policy):
    """
    Katsura: MCTS agent for Connect-4.

    Parameters
    ----------
    n_simulations : int
        Number of MCTS simulations per move. Higher = stronger but slower.
        Default: 300 (safe for Gradescope's 10s timeout).
    c_exploration : float
        UCB1 exploration constant C. √2 ≈ 1.414 is the classic value.
    rollout_policy : str
        'random'    – purely random rollouts (fast).
        'heuristic' – rollouts that immediately win or block (stronger).
    time_limit : float or None
        Hard cap in seconds per move. If set, MCTS stops early when reached.
        Default: 8.0 (leaves 2s margin under Gradescope's 10s limit).
    """

    def __init__(self,
                 n_simulations: int = 300,
                 c_exploration: float = math.sqrt(2),
                 rollout_policy: str = "heuristic",
                 time_limit: Optional[float] = 8.0):
        self.n_simulations = n_simulations
        self.c_exploration = c_exploration
        self.rollout_policy = rollout_policy
        self.time_limit = time_limit

    @override
    def mount(self, timeout=None) -> None:
        """No offline training needed – MCTS works fully online."""
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Choose a column to play.

        s : np.ndarray shape (6,7)
            −1 = Red (first player), 1 = Yellow, 0 = empty.
        """
        red_count = int(np.sum(s == -1))
        yellow_count = int(np.sum(s == 1))
        our_player = -1 if red_count == yellow_count else 1

        free = _free_cols(s)
        if len(free) == 1:
            return free[0]

        # Immediate win
        for col in free:
            for r in range(ROWS - 1, -1, -1):
                if s[r, col] == EMPTY:
                    s[r, col] = our_player
                    win = _winner_after_drop(s, r, col, our_player)
                    s[r, col] = EMPTY
                    if win:
                        return col
                    break

        # Block immediate opponent win
        opp = -our_player
        for col in free:
            for r in range(ROWS - 1, -1, -1):
                if s[r, col] == EMPTY:
                    s[r, col] = opp
                    win = _winner_after_drop(s, r, col, opp)
                    s[r, col] = EMPTY
                    if win:
                        return col
                    break

        return mcts(s, our_player, self.n_simulations,
                    self.c_exploration, self.rollout_policy,
                    self.time_limit)