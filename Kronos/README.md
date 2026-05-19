# Kronos — Connect-4 Agent

> *"El que controla el tiempo, controla el juego."*

**Estrategia:** Minimax con poda Alfa-Beta + Q-Learning online sobre pesos heurísticos  
**Diferenciador:** A diferencia de un agente de búsqueda puro, Kronos *aprende* qué rasgos del tablero son más relevantes contra cada oponente, ajustando sus pesos heurísticos durante la partida mediante TD(0).

---

## Arquitectura

```
Kronos
├── Minimax con poda α-β  ─── búsqueda del mejor movimiento
│     └── Heurística lineal: w · φ(s)
│           φ(s) = [centro, ventanas, 2-en-raya, 3-en-raya, amenaza-oponente]
└── Q-Learning online ────── actualiza w tras cada movimiento
      Δw = α · δ · ∇φ    donde δ = V(s') - V(s)
```

## Parámetros configurables ("tornillos")

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `depth` | 4 | Profundidad del minimax — más profundo = más fuerte pero más lento |
| `use_q_learning` | True | Activa/desactiva el módulo de aprendizaje online |
| `learning_rate` | 0.05 | α — qué tan rápido cambian los pesos |
| `exploration_rate` | 0.05 | ε — fracción de movimientos aleatorios (exploración) |
| `seed` | 42 | Semilla RNG para reproducibilidad |

## Uso rápido

```python
from Kronos.policy import Kronos

# Instancia para torneo (configuración final recomendada)
agent = Kronos(depth=3, use_q_learning=True, exploration_rate=0.0)
agent.mount()   # llamar antes de cada partida

# Durante la partida
col = agent.act(board)   # board: np.ndarray (6×7)
```

## Estructura de archivos

```
Kronos/
├── policy.py          ← Agente completo (este archivo es la entrega)
entrega.ipynb          ← Análisis experimental y gráficas
README.md              ← Este archivo
```

## Requisitos

```
numpy
matplotlib
```

## Resultados mínimos verificados

- ✅ Nunca pierde contra jugador aleatorio (0 derrotas en 60+ partidas por color)
- ✅ Gana ≥50% contra aleatorio (win rate observado: **100%** con depth ≥ 1)
- ✅ Diferente a agentes del grupo en al menos un aspecto conceptual (Q-Learning adaptativo)
