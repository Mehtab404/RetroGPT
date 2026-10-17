# RetroGPT Data Sources & Documentation

This directory contains source documents and knowledge base materials for classic/retro video games (Pong, Asteroids, Centipede, Missile Command, Breakout, Space Invaders, Pac-Man, etc.).

## Directory Layout
- `raw/`: Unprocessed source files (PDFs, Markdown, text, HTML manuals). Note: Binary and large PDF files are excluded from Git via `.gitignore`.

## Sources & License Registry

| Game | Title / Document | Source URL | Format | License / Terms | Notes |
|---|---|---|---|---|---|
| Asteroids | Atari Asteroids Operation & Service Manual | [Archive.org](https://archive.org/) | PDF/TXT | Public Domain / Fair Use (Vintage Manual) | Base gameplay rules & scoring |
| Pong | Atari Pong Service Manual | [Archive.org](https://archive.org/) | PDF/TXT | Public Domain / Fair Use (Vintage Manual) | Controls & mechanics |
| Centipede | Atari Centipede Game Manual | [Archive.org](https://archive.org/) | PDF/TXT | Public Domain / Fair Use (Vintage Manual) | Scoring & insect behavior |
| Missile Command | Atari Missile Command Manual | [Archive.org](https://archive.org/) | PDF/TXT | Public Domain / Fair Use (Vintage Manual) | Cities, silos, & scoring |
| Breakout | Atari Breakout Operations Manual | [Archive.org](https://archive.org/) | PDF/TXT | Public Domain / Fair Use (Vintage Manual) | Brick rows, angles, scoring |

## Guidelines for Adding Data
1. Ensure all documents are public or permissible for research/evaluation under fair use.
2. Store documents under `data/raw/<game_name>/` or `data/raw/<document_name>`.
3. Update this table whenever new game manuals or knowledge articles are added.
