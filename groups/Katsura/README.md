# Katsura 🌸 — MCTS Agent for Connect-4

**Author:** Nicolas Esteban Muñoz Sendoya 
**Course:** Fundamentos de Inteligencia Artificial — Universidad de La Sabana, 2026.1

---

## Idea principal

Katsura implementa **Monte-Carlo Tree Search (MCTS)** con exploración **UCB1**, el algoritmo de búsqueda en árbol presentado en la Clase 13 (*Online Policy Improvement*) y con la fórmula de exploración basada en probabilidades de victoria de la Clase 12 (*Competitive MDPs*).

En lugar de evaluar el tablero con una función heurística fija (como haría Minimax), Katsura **simula miles de partidas aleatorias** desde el estado actual y acumula estadísticas de victorias y visitas en un árbol. La acción elegida es la columna con más visitas, que estadísticamente corresponde a la mejor jugada.

### Las cuatro fases de MCTS por movimiento

```
Selection → Expansion → Rollout → Backpropagation
```

1. **Selection:** Se recorre el árbol con UCB1 hasta un nodo no completamente expandido.
2. **Expansion:** Se añade un hijo nuevo (acción no explorada aún).
3. **Rollout:** Se juega la partida hasta el final con política aleatoria (o heurística).
4. **Backpropagation:** El resultado se propaga hacia arriba, cambiando de signo en cada nivel (juego de suma cero).

### Tornillos de configuración

| Parámetro | Por defecto | Efecto |
|---|---|---|
| `n_simulations` | 500 | Presupuesto de simulaciones por movimiento. Más → más fuerte, más lento. |
| `c_exploration` | √2 ≈ 1.414 | Constante UCB1. Mayor → más exploración. |
| `rollout_policy` | `'heuristic'` | `'random'`: rollouts puramente aleatorios. `'heuristic'`: prefiere ganar/bloquear inmediatamente. |

---

## Estructura de archivos

```
katsura/
├── policy.py    ← Agente completo (clase Katsura + lógica MCTS interna)
└── readme.md    ← Este archivo
```

El agente es **self-contained**: no tiene dependencias externas más allá de `numpy` y la librería estándar de Python.

---

## Cómo usar el agente

### Dentro del torneo

Copiar la carpeta `katsura/` dentro de `tournament/groups/` y ejecutar:

```bash
cd tournament/
python main.py
```

El framework detecta automáticamente la clase `Katsura` que extiende `Policy`.

### Uso standalone (para el notebook de análisis)

```python
import numpy as np
from policy import Katsura

agent = Katsura(n_simulations=300, rollout_policy="heuristic")
agent.mount()  # sin entrenamiento offline

board = np.zeros((6, 7), dtype=int)
col = agent.act(board)
print(f"Katsura juega en columna {col}")
```

### Ajustar parámetros para el análisis

```python
# Versión rápida / bajo recurso
agent_light = Katsura(n_simulations=50, rollout_policy="random")

# Versión estándar
agent_mid   = Katsura(n_simulations=200, rollout_policy="heuristic")

# Versión fuerte / alto recurso
agent_heavy = Katsura(n_simulations=800, rollout_policy="heuristic")
```

---

## Diferencia conceptual respecto a otros agentes

Katsura no requiere ni función de evaluación manual ni tabla de Q-values preentrenada.  
Su fortaleza viene exclusivamente de **simulación masiva + estadística**, lo que lo hace adaptable a cualquier oponente sin reentrenamiento.

---

## Propuestas de mejora futura

- **Inicialización de Q-values con red neuronal** (AlphaZero-style): usar una red que guíe tanto la selección como los rollouts.
- **Transposition table**: almacenar nodos MCTS para estados ya visitados en partidas anteriores (reutilización entre jugadas).
- **Rollouts más inteligentes**: incorporar patrones de amenaza más complejos (traps de dos en fila, zugzwang).
