# Kronos — Agente Connect-4

**Agente desarrollado para el reto de Fundamentos de Inteligencia Artificial, Universidad de La Sabana, 2026.1**

---

## Idea principal

Kronos es un agente basado en **Alternating Markov Game** que aprende a jugar Connect-4 mediante **self-play con First-Visit Monte Carlo (FVMC)** y **exploración proporcional a q̂** (win-probability-guided exploration). 

### Fundamentos teóricos 

| Concepto | Aplicación en Kronos |
|---|---|---|
| Alternating Markov Game | Self-play: ambos jugadores comparten y sincronizan la Q-table en cada trial |
| First-Visit Monte Carlo (FVMC) | Estima q̂(s,a) con retornos descontados de la primera visita a cada (s,a) |
| Exploración proporcional a q̂ | π(a\|s) = q̂(s,a) / Σ q̂(s,a') — focaliza exploración en estados ganadores |
| Reward Shaping | Recompensa intermedia por amenazas de 3-en-raya para guiar aprendizaje |
| Simetría horizontal | Canonicaliza tablero (min de original y espejo) → espacio de estados ÷2 |

### Lo que distingue a Kronos

- **Self-play sincronizado** (Alternating Markov Game): 1 trial actualiza ambos jugadores simultáneamente, no k trials para k agentes.
- **Simetría de tablero**: reduce el espacio de estados efectivo a la mitad sin pérdida de información.
- **Aprendizaje online**: al terminar cada partida real, actualiza q̂ con el resultado verdadero (tasa α).

---

## Estructura de archivos

```
kronos/
├── policy.py          # Clase Kronos (política principal)
├── kronos_q.pkl       # Q-table entrenada (se genera al primer mount())
├── entrega.ipynb      # Notebook con experimentos y gráficas
└── README.md          # Este archivo
```

---

## Uso

```python
from policy import Kronos

agente = Kronos(
    episodes=6000,          # episodios de self-play offline
    shaping_weight=0.05,    # peso del reward shaping
    exploration_temp=1.0,   # temperatura de exploración
    online_alpha=0.05,      # aprendizaje online
)
agente.mount()              # carga o entrena la Q-table
col = agente.act(board)     # juega
agente.observe_result(winner)  # actualización online (opcional)
```

### Variantes configurables

| Parámetro | Descripción | Efecto esperado |
|---|---|---|
| `episodes` | Episodios de entrenamiento offline | Más → mejor desempeño, más tiempo |
| `shaping_weight` | Peso del reward shaping | Más → más agresivo en ataque/bloqueo |
| `exploration_temp` | Temperatura softmax | Más baja → más greedy durante training |
| `online_alpha` | Tasa aprendizaje online | Más alto → aprende más rápido del oponente |

---

## Requisitos

```
numpy
pickle (stdlib)
```

---

## Enlace al repositorio

> *(https://github.com/N1kxo/agents-connect4-munoz_sarmiento)*
