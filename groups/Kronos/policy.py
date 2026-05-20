"""
Kronos - Agente Connect-4

"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict


import numpy as np
from connect4.policy import Policy

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto (se pueden sobreescribir al instanciar)
# ---------------------------------------------------------------------------
DEFAULT_SELF_PLAY_EPISODES = 6_000   # trials de entrenamiento offline
DEFAULT_GAMMA = 1.0                  # juego de suma cero: sólo importa el final
DEFAULT_REWARD_WIN = 1.0
DEFAULT_REWARD_LOSS = -1.0
DEFAULT_REWARD_DRAW = 0.0
DEFAULT_SHAPING_WEIGHT = 0.05        # peso de la recompensa de shaping
DEFAULT_EXPLORATION_TEMP = 1.0       # temperatura para softmax de exploración
DEFAULT_ONLINE_ALPHA = 0.05          # tasa de aprendizaje online (Exploring Starts)
SAVE_PATH = os.path.join(os.path.dirname(__file__), "kronos_q.pkl")

ROWS, COLS = 6, 7


# ---------------------------------------------------------------------------
# Utilidades de estado
# ---------------------------------------------------------------------------

def board_to_key(board: np.ndarray, player: int) -> tuple:
    """
    Convierte el tablero a una clave hashable canónica.
    Aplica simetría horizontal: usa el mínimo entre el tablero original
    y su espejo para reducir el espacio de estados.
    """
    flat = board.flatten().tolist()
    mirror = board[:, ::-1].flatten().tolist()
    canonical = min(flat, mirror)
    return tuple(canonical) + (player,)


def get_free_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    for r in reversed(range(ROWS)):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def check_winner(board: np.ndarray) -> int:
    """Devuelve -1, 1 o 0."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r, c]
            if p == 0:
                continue
            # Horizontal
            if c + 3 < COLS and all(board[r, c + i] == p for i in range(4)):
                return p
            # Vertical
            if r + 3 < ROWS and all(board[r + i, c] == p for i in range(4)):
                return p
            # Diagonal ↘
            if r + 3 < ROWS and c + 3 < COLS and all(
                board[r + i, c + i] == p for i in range(4)
            ):
                return p
            # Diagonal ↙
            if r + 3 < ROWS and c - 3 >= 0 and all(
                board[r + i, c - i] == p for i in range(4)
            ):
                return p
    return 0


def is_terminal(board: np.ndarray) -> bool:
    return check_winner(board) != 0 or not any(board[0] == 0)


# ---------------------------------------------------------------------------
# Reward shaping: recompensa intermedia por amenazas
# ---------------------------------------------------------------------------

def shaping_reward(board: np.ndarray, player: int) -> float:
    """
    Reward shaping (slide 11): añade señal intermedia contando ventanas de 3
    fichas propias sin bloquear (potenciales victorias). Ayuda al agente a
    aprender estrategia sin esperar a la recompensa terminal escasa.
    """
    score = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                window = []
                for i in range(4):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        window.append(board[nr, nc])
                    else:
                        window.append(None)
                if None in window:
                    continue
                own = window.count(player)
                opp = window.count(-player)
                empty = window.count(0)
                if opp == 0 and own == 3 and empty == 1:
                    score += 1.0
                elif opp == 3 and own == 0 and empty == 1:
                    score -= 1.0  # penalizar amenazas del oponente
    return score


# ---------------------------------------------------------------------------
# Tabla-Q compartida (Alternating Markov Game)
# ---------------------------------------------------------------------------

class SharedQTable:
    """
    Tabla q̂(s, a) compartida entre ambos jugadores del Alternating Markov Game.
    El estado incluye el jugador activo, por lo que la misma tabla sirve para ambos.
    El valor q̂(s,a) representa la probabilidad estimada de ganar desde (s,a).
    """

    def __init__(self):
        self.q: dict[tuple, dict[int, float]] = defaultdict(dict)
        self.n: dict[tuple, dict[int, int]] = defaultdict(dict)

    def get(self, key: tuple, col: int) -> float:
        return self.q[key].get(col, 0.5)  # prior neutral: 50% de ganar

    def update(self, key: tuple, col: int, value: float):
        if col not in self.n[key]:
            self.n[key][col] = 0
            self.q[key][col] = 0.5
        self.n[key][col] += 1
        # Actualización incremental (FVMC)
        self.q[key][col] += (value - self.q[key][col]) / self.n[key][col]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q": dict(self.q), "n": dict(self.n)}, f)

    @classmethod
    def load(cls, path: str) -> "SharedQTable":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.q = defaultdict(dict, data["q"])
        obj.n = defaultdict(dict, data.get("n", {}))
        return obj


# ---------------------------------------------------------------------------
# Entrenamiento: Self-play con Alternating Markov Game + FVMC
# ---------------------------------------------------------------------------

def softmax_probs(values: list[float], temp: float) -> list[float]:
    """
    Distribución proporcional a q̂ (slide 12).
    π(a|s) = q̂(s,a) / Σ q̂(s,a')
    Con temperatura para controlar la exploración.
    """
    vals = np.array(values)
    # Escalar con temperatura
    vals = vals / max(temp, 1e-6)
    vals = vals - vals.max()  # estabilidad numérica
    exp_vals = np.exp(vals)
    total = exp_vals.sum()
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return (exp_vals / total).tolist()


def run_self_play_episode(
    q_table: SharedQTable,
    shaping_weight: float = DEFAULT_SHAPING_WEIGHT,
    exploration_temp: float = DEFAULT_EXPLORATION_TEMP,
) -> list[tuple[tuple, int, float]]:
    """
    Genera un trial de self-play (Alternating Markov Game, slide 12).
    Ambos jugadores comparten la misma q_table y alternan turnos.
    Retorna la historia de (key, col, retorno) para actualizar con FVMC.
    """
    board = np.zeros((ROWS, COLS), dtype=int)
    player = -1  # Rojo empieza
    trajectory: list[tuple[tuple, int, float, int]] = []
    # (key, col, shaping_r, player_who_moved)

    while not is_terminal(board):
        free_cols = get_free_cols(board)
        key = board_to_key(board, player)

        # Política exploratoria: proporcional a q̂ (slide 12)
        q_vals = [q_table.get(key, c) for c in free_cols]
        probs = softmax_probs(q_vals, exploration_temp)
        col = int(np.random.choice(free_cols, p=probs))

        # Reward shaping antes de la transición
        new_board = apply_move(board, col, player)
        sr = shaping_weight * shaping_reward(new_board, player)

        trajectory.append((key, col, sr, player))
        board = new_board
        player = -player

    # Recompensa terminal
    winner = check_winner(board)

    # FVMC: propagar retornos hacia atrás (γ=1)
    # Alternating Markov Game (slide 12): multiplicar por -1 al cambiar de jugador
    updates: list[tuple[tuple, int, float]] = []
    G = 0.0
    last_player = -player  # el jugador que hizo el último movimiento

    if winner == 0:
        terminal_reward = DEFAULT_REWARD_DRAW
    elif winner == last_player:
        terminal_reward = DEFAULT_REWARD_WIN
    else:
        terminal_reward = DEFAULT_REWARD_LOSS

    G = terminal_reward

    # Primera visita: registrar las primeras ocurrencias de (key, col)
    seen: set[tuple[tuple, int]] = set()
    rev_traj = list(reversed(trajectory))

    for i, (key, col, sr, mover) in enumerate(rev_traj):
        # Invertir el retorno al cambiar de jugador (juego de suma cero, slide 12)
        if i > 0 and rev_traj[i - 1][3] != mover:
            G = -G
        G += sr  # añadir reward shaping
        pair = (key, col)
        if pair not in seen:
            seen.add(pair)
            updates.append((key, col, G))

    return updates


def train_kronos(
    episodes: int = DEFAULT_SELF_PLAY_EPISODES,
    shaping_weight: float = DEFAULT_SHAPING_WEIGHT,
    exploration_temp: float = DEFAULT_EXPLORATION_TEMP,
    save_path: str = SAVE_PATH,
    verbose: bool = False,
) -> SharedQTable:
    """
    Entrenamiento offline de Kronos mediante self-play.
    Implementa Alternating Markov Game + FVMC (slides 11 y 12).
    """
    q_table = SharedQTable()
    for ep in range(episodes):
        updates = run_self_play_episode(q_table, shaping_weight, exploration_temp)
        for key, col, G in updates:
            q_table.update(key, col, G)
        if verbose and (ep + 1) % 500 == 0:
            print(f"  Episodio {ep + 1}/{episodes} completado.")
    if save_path:
        q_table.save(save_path)
        if verbose:
            print(f"  Q-table guardada en {save_path}")
    return q_table


# ---------------------------------------------------------------------------
# Política de juego (online)
# ---------------------------------------------------------------------------

class Kronos(Policy):
    """
    Kronos: agente Connect-4 basado en Alternating Markov Game.

    Parámetros
    ----------
    episodes : int
        Número de episodios de self-play para entrenamiento offline.
    shaping_weight : float
        Peso de la recompensa de shaping por amenazas.
    exploration_temp : float
        Temperatura de exploración (softmax sobre q̂).
    online_alpha : float
        Tasa de aprendizaje para actualización online durante el juego real.
    force_retrain : bool
        Si True, ignora la Q-table guardada y reentrena desde cero.
    """

    def __init__(
        self,
        episodes: int = DEFAULT_SELF_PLAY_EPISODES,
        shaping_weight: float = DEFAULT_SHAPING_WEIGHT,
        exploration_temp: float = DEFAULT_EXPLORATION_TEMP,
        online_alpha: float = DEFAULT_ONLINE_ALPHA,
        force_retrain: bool = False,
    ):
        self.episodes = episodes
        self.shaping_weight = shaping_weight
        self.exploration_temp = exploration_temp
        self.online_alpha = online_alpha
        self.force_retrain = force_retrain
        self.q_table: SharedQTable | None = None
        # Historial del juego en curso (para aprendizaje online)
        self._game_history: list[tuple[tuple, int]] = []
        self._my_player: int | None = None

    
    def mount(self) -> None:
        """
        Carga o entrena la Q-table. Se llama una vez antes del torneo.
        """
        if not self.force_retrain and os.path.exists(SAVE_PATH):
            self.q_table = SharedQTable.load(SAVE_PATH)
        else:
            self.q_table = train_kronos(
                episodes=self.episodes,
                shaping_weight=self.shaping_weight,
                exploration_temp=self.exploration_temp,
                save_path=SAVE_PATH,
            )
        self._game_history = []
        self._my_player = None

    def act(self, s: np.ndarray) -> int:
        """
        Selecciona la acción greedy según q̂(s,a) (slide 12: arg max q̂).
        Si hay empate en q̂, prefiere columnas centrales (heurística).
        Realiza actualización online ligera (Exploring Starts, slide 11).
        """
        assert self.q_table is not None, "Debes llamar mount() antes de act()."

        # Inferir el jugador activo desde el tablero
        red_count = int((s == -1).sum())
        yellow_count = int((s == 1).sum())
        player = -1 if red_count == yellow_count else 1

        if self._my_player is None:
            self._my_player = player

        free_cols = get_free_cols(s)
        key = board_to_key(s, player)

        # --- Detección de movida ganadora inmediata (regla hard-coded) ---
        for col in free_cols:
            test = apply_move(s, col, player)
            if check_winner(test) == player:
                self._game_history.append((key, col))
                return col

        # --- Bloquear victoria inmediata del oponente ---
        for col in free_cols:
            test = apply_move(s, col, -player)
            if check_winner(test) == -player:
                self._game_history.append((key, col))
                return col

        # --- Selección greedy por q̂ con desempate central ---
        center_bonus = {c: -abs(c - 3) * 0.001 for c in range(COLS)}
        best_col = max(
            free_cols,
            key=lambda c: self.q_table.get(key, c) + center_bonus[c],
        )

        self._game_history.append((key, best_col))
        return best_col

    def observe_result(self, winner: int) -> None:
        """
        Llamar al final de la partida para aprendizaje online (FVMC).
        Actualiza q̂ con el resultado real de la partida.
        """
        if self.q_table is None or self._my_player is None:
            return
        if winner == self._my_player:
            G = DEFAULT_REWARD_WIN
        elif winner == 0:
            G = DEFAULT_REWARD_DRAW
        else:
            G = DEFAULT_REWARD_LOSS

        # Propagar retorno hacia atrás (FVMC, γ=1)
        seen: set[tuple[tuple, int]] = set()
        for key, col in reversed(self._game_history):
            pair = (key, col)
            if pair not in seen:
                seen.add(pair)
                old = self.q_table.get(key, col)
                # Actualización con alpha fija (online)
                self.q_table.q[key][col] = old + self.online_alpha * (G - old)
        self._game_history = []
        self._my_player = None
