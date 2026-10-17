# RetroGPT Data Sources & Documentation

This directory contains source documents and knowledge base materials for classic/retro video games (Pong, Asteroids, Centipede, Missile Command, Breakout, Space Invaders, Pac-Man, etc.).

## Directory Layout
- `raw/`: Unprocessed source files (PDFs, Markdown, text, HTML manuals). Note: Binary and large PDF files are excluded from Git via `.gitignore`.

## Sources & License Registry

All game manuals are created based on publicly available information about classic arcade games. These documents are for educational and portfolio demonstration purposes under fair use.

| Game | Document | Format | Date Added | Word Count | Notes |
|---|---|---|---|---|---|
| Asteroids | asteroids_manual.md | Markdown | 2026-10-17 | ~800 | Complete gameplay, scoring, controls |
| Pac-Man | pacman_manual.md | Markdown | 2026-10-17 | ~1200 | Ghost AI, scoring, fruit progression |
| Space Invaders | space_invaders_manual.md | Markdown | 2026-10-17 | ~1100 | UFO patterns, scoring, strategy |
| Centipede | centipede_manual.md | Markdown | 2026-10-17 | ~1300 | Multi-enemy mechanics, mushroom field |
| Missile Command | missile_command_manual.md | Markdown | 2026-10-17 | ~1600 | Defense strategy, battery management |
| Breakout | breakout_manual.md | Markdown | 2026-10-17 | ~1000 | Paddle control, brick breaking, tunneling |

### Content Sources
Information compiled from:
- Wikipedia articles on classic arcade games
- StrategyWiki game guides
- Retro gaming community documentation
- Original arcade game observations

### License
These documents are created for educational purposes as part of a portfolio RAG system demonstration. They contain factual information about classic video games that is widely available in the public domain.

## Guidelines for Adding Data
1. Ensure all documents are public or permissible for research/evaluation under fair use.
2. Store documents under `data/raw/<game_name>/` or `data/raw/<document_name>`.
3. Update this table whenever new game manuals or knowledge articles are added.
