Pacman vs Ghosts — AI vs AI (A*/GBFS + Minimax + Fuzzy Logic)

An AI-driven Pacman game where Pacman and two Ghosts battle using a hybrid of fuzzy logic, A*/GBFS pathfinding, and minimax with alpha–beta pruning. Built with Python and Pygame.



Table of Contents
- Overview
- Features
- Tech Stack
- Project Structure
- Getting Started
- Controls
- Configuration
- How It Works (AI/Algorithms)
- Sequence Diagram (Game Loop)
- Architecture Diagram
- Scoring and Mechanics
- Troubleshooting


Overview
This project procedurally generates a maze and pits an autonomous Pacman against two autonomous Ghosts:
- Pacman uses fuzzy logic to pick a high-level mode (Eat Food, Eat Power, Evade), then plans paths with A* or GBFS.
- Ghosts use a fuzzy behavior selector to switch strategies (Aggressive, Coordinated Hunt, Intercept, Scatter, Defensive) and a depth-limited minimax to choose moves.


Features
- Procedural maze generation (19×19, configurable)
- Hybrid AI for Pacman: Fuzzy Logic + A*/GBFS
- Hybrid AI for Ghosts: Fuzzy Logic + Minimax (alpha–beta)
- Dynamic spawning of power pellets to balance gameplay
- Clean rendering with animated Pacman mouth and ghost bobbing/eyes
- HUD with step counter, score, lives, and current ghost behaviors


Tech Stack
- Python 3.9+
- Pygame


Project Structure

```text
ai/
├── pacman.py     # Main game, renderer, Pacman agent, procedural maze
├── ghosts.py     # Ghost AI (fuzzy behavior + minimax) and pathfinder
└── __pycache__/  # Python bytecode cache
```


Getting Started

1) Install Python and dependencies

```bash
python -m pip install --upgrade pip
pip install pygame
```

If you want a pinned environment, create a virtual environment first:

```bash
python -m venv .venv
./.venv/Scripts/activate        # Windows
# source .venv/bin/activate     # macOS/Linux
pip install pygame
```

2) Run

```bash
python pacman.py
```

The game launches in fullscreen. It auto-generates a fresh maze each run.


Controls
- ESC: Exit the game (and the game-over screen)
- SPACE: Close game-over screen (also exits)


Configuration
Key parameters in `pacman.py` you may tweak:

```python
TILE = 32                 # base tile size; auto-scales for screen
FPS = 15                  # target frames per second
MAX_GAME_STEPS = 3000     # end condition if exceeded
GHOST_MINIMAX_DEPTH = 1   # ghost minimax search depth
PACMAN_REPLAN_STEPS = 3   # how often Pacman replans path
```

Maze size defaults to 19×19 but can be changed by passing different values to `generate_random_maze(width, height)`.


How It Works (AI/Algorithms)

- Pacman
  - Fuzzy Logic (`PacmanFuzzy`): computes degrees for “close ghost”, “far ghost”, “near power”, “low food”, then blends rules to pick a mode: `EAT_FOOD`, `EAT_POWER`, or `EVADE`.
  - Pathfinding (`Pathfinder`): Uses A* to reach goals (food/power/escape tiles) and falls back to GBFS when needed. Adds penalties for recently traversed tiles to avoid loops when stuck.

- Ghosts
  - Fuzzy Behavior (`GhostFuzzy`): evaluates distances, threat (power pellets/scared timer), coordination, and food context to choose a high-level behavior: `AGGRESSIVE_PURSUIT`, `COORDINATED_HUNT`, `INTERCEPT`, `SCATTER`, `DEFENSIVE`.
  - Minimax (`GhostMinimax`): depth-limited alpha–beta over ghost move (max) vs. pacman move (min), with a heuristic that incorporates the chosen behavior’s modifiers, spacing, stuck penalties, and intercept opportunities.

- Game Mechanics
  - Dynamic power pellets spawn mid-game to reduce difficulty spikes (e.g., after deaths or long droughts since last power), with safe distance constraints from ghosts.
  - Scared timer counts down once per full turn (after the last ghost moves).


Sequence Diagram (Game Loop)

```mermaid
sequenceDiagram
    participant Loop as Game.run()
    participant Pac as PacmanAgent
    participant Ghost0 as GhostAgent0
    participant Ghost1 as GhostAgent1
    participant R as Renderer

    Loop->>Pac: choose_action(state)
    Pac-->>Loop: (next_pos, mode, path_overlay)
    Loop->>Loop: apply_pac_move(state, next_pos)

    Loop->>Ghost0: choose_action(state)
    Ghost0-->>Loop: move0
    Loop->>Loop: apply_ghost_move(state, move0, 0)

    Loop->>Ghost1: choose_action(state)
    Ghost1-->>Loop: move1
    Loop->>Loop: apply_ghost_move(state, move1, 1)

    Loop->>Loop: spawn_power_pellet_if_needed()
    Loop->>R: draw_grid(state, overlay)
    Loop->>R: draw_hud(steps, mode, score, lives, behaviors)
    R-->>Loop: frame rendered
```


Architecture Diagram (High-Level)

```mermaid
flowchart LR
    subgraph Game
        G[Game.run / update_logic]
        R[Renderer]
    end

    subgraph Pacman
        PF[Fuzzy (mode)]
        PP[Pathfinder (A*/GBFS)]
        PA[PacmanAgent]
    end

    subgraph Ghosts
        GF[Fuzzy (behavior)]
        MM[Minimax + Alpha-Beta]
        GA0[GhostAgent[0]]
        GA1[GhostAgent[1]]
    end

    S[(State)]

    G --> PA
    PA --> PF
    PA --> PP
    PF --> PA
    PP --> PA
    PA --> S

    G --> GA0
    G --> GA1
    GA0 --> GF
    GA0 --> MM
    GA1 --> GF
    GA1 --> MM
    GA0 --> S
    GA1 --> S

    G --> R
    S --> R
```


Scoring and Mechanics
- Pellet eaten: +50
- Power pellet: +500 and set `scared_timer = 240`
- Ghost eaten while scared: +750 base, +1500 bonus (total +2250)
- Pacman death: −200, lose a life, brief invulnerability on respawn
- Win/Lose/End conditions:
  - Win: All pellets eaten or score ≥ 30,000 (and lives > 0)
  - Lose: Lives ≤ 0
  - End: Step counter ≥ `MAX_GAME_STEPS`


Troubleshooting
- Black screen or crash on start
  - Ensure `pygame` is installed: `pip show pygame`
  - Try windowed mode by changing the display flag in `pacman.py`:
    ```python
    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))  # remove pygame.FULLSCREEN
    ```
- Very large or ultra-wide monitors
  - The game auto-scales, but you can cap `self.tile_size` (already capped to 48) or adjust HUD height.
- Performance
  - Lower `FPS` or simplify animations; reduce ghost minimax depth (`GHOST_MINIMAX_DEPTH`).


---

If you use this for teaching or demos, consider adding your own demo GIF under `docs/` and linking it at the top.


