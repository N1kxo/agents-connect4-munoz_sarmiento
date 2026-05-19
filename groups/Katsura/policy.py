"""
Katsura – MCTS Agent for Connect-4
===================================
Monte-Carlo Tree Search with UCB1 exploration for Connect-4.

Based on concepts from:
  - Slide 13 (Online Policy Improvement / MCTS)
  - Slide 12 (Competitive MDPs, UCB-based exploration in Bernoulli games)

How it works
------------
On every call to act(), Katsura runs `n_simulations` MCTS episodes from the
current board state.  Each episode has four phases:

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
      'heuristic' rollouts prefer winning moves and blocking moves, making
      each simulation more informative at the cost of a little speed.
"""

import math
import random
import numpy as np
from abc import ABC
from typing import Optional, List

# ---------------------------------------------------------------------------
# Python 3.10 compatibility: override decorator and typing annotations
# ---------------------------------------------------------------------------

try:
    from typing import override
except ImportError:
    def override(f): return f


# ---------------------------------------------------------------------------
# Import the base class the tournament framework provides
# ---------------------------------------------------------------------------
try:
    from connect4.policy import Policy
except ImportError:
    # Fallback so the file can be read standalone (e.g. in the notebook)
    class Policy(ABC):
        def mount(self): pass
        def act(self, s): pass


# ============================================================================
# Internal game logic (self-contained, no dependency on ConnectState object)
# ============================================================================

ROWS = 6
COLS = 7
EMPTY = 0


def _drop(board: np.ndarray, col: int, player: int) -> Optional[np.ndarray]:
    """Return a new board after dropping `player`'s piece in `col`, or None if invalid."""
    if board[0, col] != EMPTY:
        return None
    new = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if new[r, col] == EMPTY:
            new[r, col] = player
            break
    return new


def _free_cols(board: np.ndarray) -> List[int]:
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def _winner(board: np.ndarray) -> int:
    """Return winning player (−1 or 1) or 0 if none."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            # right
            if c + 3 < COLS and all(board[r, c + i] == p for i in range(4)):
                return p
            # down
            if r + 3 < ROWS and all(board[r + i, c] == p for i in range(4)):
                return p
            # diag right-down
            if r + 3 < ROWS and c + 3 < COLS and all(board[r + i, c + i] == p for i in range(4)):
                return p
            # diag left-down
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r + i, c - i] == p for i in range(4)):
                return p
    return 0


def _is_terminal(board: np.ndarray) -> bool:
    return _winner(board) != 0 or len(_free_cols(board)) == 0


def _heuristic_move(board: np.ndarray, player: int) -> int | None:
    """
    Quick look-ahead heuristic for rollouts:
      1. If we can win immediately, do it.
      2. If the opponent can win immediately, block it.
      3. Otherwise return None (fall back to random).
    """
    cols = _free_cols(board)
    # Check win
    for c in cols:
        b = _drop(board, c, player)
        if b is not None and _winner(b) == player:
            return c
    # Check block
    opp = -player
    for c in cols:
        b = _drop(board, c, opp)
        if b is not None and _winner(b) == opp:
            return c
    return None


# ============================================================================
# MCTS Node
# ============================================================================

class _Node:
    __slots__ = ("board", "player", "parent", "action",
                 "children", "untried", "wins", "visits")

    def __init__(self, board: np.ndarray, player: int,
                 parent: Optional["_Node"]= None, action: Optional[int] = None):
        self.board = board
        self.player = player          # the player whose turn it is at this node
        self.parent = parent
        self.action = action          # action that led from parent to this node
        self.children: List["_Node"] = []
        self.untried: List[int] = _free_cols(board)
        random.shuffle(self.untried)  # randomise expansion order

        self.wins: float = 0.0
        self.visits: int = 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return _is_terminal(self.board)

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
        """Expand one untried action and return the new child node."""
        col = self.untried.pop()
        new_board = _drop(self.board, col, self.player)
        child = _Node(new_board, -self.player, parent=self, action=col)
        self.children.append(child)
        return child

    def backpropagate(self, result: float, root_player: int) -> None:
        """Walk up the tree updating wins/visits.
        `result` is +1 if root_player won, -1 if lost, 0 if draw."""
        node = self
        while node is not None:
            node.visits += 1
            # From the perspective of root_player:
            # even-depth nodes are root_player's turns → positive result is good
            # odd-depth nodes are opponent's turns  → positive result is bad
            if node.player != root_player:
                node.wins += result
            else:
                node.wins -= result
            node = node.parent


# ============================================================================
# MCTS runner
# ============================================================================

def _rollout(board: np.ndarray, player: int, policy: str) -> int:
    """Simulate a random game from `board` and return the winner (or 0)."""
    current_player = player
    b = board.copy()
    while not _is_terminal(b):
        if policy == "heuristic":
            col = _heuristic_move(b, current_player)
            if col is None:
                col = random.choice(_free_cols(b))
        else:
            col = random.choice(_free_cols(b))
        b = _drop(b, col, current_player)
        current_player = -current_player
    return _winner(b)


def mcts(board: np.ndarray, root_player: int,
         n_simulations: int, c_exploration: float,
         rollout_policy: str) -> int:
    """
    Run MCTS from `board` for `root_player`.
    Returns the column with the highest visit count.
    """
    root = _Node(board, root_player)

    for _ in range(n_simulations):
        # --- 1. Selection ---
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(c_exploration)

        # --- 2. Expansion ---
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # --- 3. Rollout ---
        winner = _rollout(node.board, node.player, rollout_policy)

        # --- 4. Backpropagation ---
        if winner == root_player:
            result = 1.0
        elif winner == 0:
            result = 0.0
        else:
            result = -1.0
        node.backpropagate(result, root_player)

    # Pick the child with the most visits (most robust)
    best = max(root.children, key=lambda n: n.visits)
    return best.action


# ============================================================================
# Policy class — this is what the tournament framework loads
# ============================================================================

class Katsura(Policy):
    """
    Katsura: MCTS agent for Connect-4.

    Parameters
    ----------
    n_simulations : int
        Number of MCTS simulations per move.  Higher = stronger but slower.
        Default: 500  (safe for the tournament's time limits).
    c_exploration : float
        UCB1 exploration constant C.  √2 ≈ 1.414 is the classic value.
    rollout_policy : str
        'random'    – purely random rollouts (fast).
        'heuristic' – rollouts that immediately win or block (slightly stronger).
    """

    def __init__(self,
                 n_simulations: int = 500,
                 c_exploration: float = math.sqrt(2),
                 rollout_policy: str = "heuristic"):
        self.n_simulations = n_simulations
        self.c_exploration = c_exploration
        self.rollout_policy = rollout_policy

    @override
    def mount(self, timeout=None) -> None:
        """No offline training needed – MCTS works fully online."""
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Choose a column to play.

        Parameters
        ----------
        s : np.ndarray, shape (6, 7)
            Current board.  −1 = Red (first player), 1 = Yellow, 0 = empty.
            The tournament framework always passes the board from the
            perspective of the active player, so we detect our color from
            the piece count.
        """
        # Determine which player we are:
        # Red (−1) plays first so has equal or one more piece than Yellow (1).
        red_count = int(np.sum(s == -1))
        yellow_count = int(np.sum(s == 1))
        # If counts are equal it's Red's turn; otherwise Yellow's.
        our_player = -1 if red_count == yellow_count else 1

        free = _free_cols(s)
        if len(free) == 1:
            return free[0]

        # Safety: immediate win
        for col in free:
            b = _drop(s, col, our_player)
            if b is not None and _winner(b) == our_player:
                return col

        # Safety: block immediate opponent win
        opp = -our_player
        for col in free:
            b = _drop(s, col, opp)
            if b is not None and _winner(b) == opp:
                return col

        # MCTS for everything else
        return mcts(s, our_player, self.n_simulations,
                    self.c_exploration, self.rollout_policy)