# PAC-MAN - Game Manual

## Game Overview
Pac-Man is a maze action game developed by Namco and released in 1980. Guide Pac-Man through a maze, eating all dots while avoiding four colored ghosts.

## Controls
- **Joystick/Arrow Keys**: Move Pac-Man up, down, left, or right through the maze

## Gameplay

### Objective
Eat all 240 dots in the maze to complete the level while avoiding ghosts. Achieve the highest score possible.

### Dots and Pellets
- **Dots**: 10 points each, 240 dots per level
- **Power Pellets**: 50 points each, 4 per level located in maze corners
- **Bonus Fruits**: Appear twice per level, points vary by level

### Ghosts
Four ghosts chase Pac-Man with different behaviors:

**Blinky (Red)** - "Shadow"
- Directly chases Pac-Man
- Becomes faster ("Cruise Elroy") when few dots remain

**Pinky (Pink)** - "Speedy"  
- Tries to position ahead of Pac-Man
- Aims for 4 tiles in front of Pac-Man's direction

**Inky (Cyan)** - "Bashful"
- Complex behavior based on Blinky and Pac-Man's positions
- Most unpredictable ghost

**Clyde (Orange)** - "Pokey"
- Chases Pac-Man when far away
- Retreats to corner when close

### Power Pellets
When Pac-Man eats a power pellet:
- Ghosts turn blue and flee for 6-10 seconds (time decreases with level)
- Blue ghosts can be eaten for points
- Ghost points: 1st = 200, 2nd = 400, 3rd = 800, 4th = 1600
- Eaten ghosts return to center box as eyes and regenerate

### Bonus Fruits
Fruits appear below the ghost box twice per level:
- Level 1: Cherry - 100 points
- Level 2: Strawberry - 300 points
- Level 3-4: Orange - 500 points
- Level 5-6: Apple - 700 points
- Level 7-8: Melon - 1000 points
- Level 9-10: Galaxian - 2000 points
- Level 11-12: Bell - 3000 points
- Level 13+: Key - 5000 points

## Scoring System
- Dot: 10 points
- Power Pellet: 50 points
- 1st Ghost: 200 points
- 2nd Ghost: 400 points
- 3rd Ghost: 800 points
- 4th Ghost: 1600 points
- Bonus Fruit: 100-5000 points (level dependent)
- Extra life at 10,000 points

## Maze Navigation
- Pac-Man moves through corridors
- Two tunnels on sides connect left and right edges
- Ghosts slow down in tunnels - use for escape
- Cannot reverse direction instantly - must turn corners

## Ghost AI Modes
Ghosts alternate between three modes:

**Chase Mode**: Ghosts actively pursue Pac-Man
**Scatter Mode**: Ghosts retreat to home corners
**Frightened Mode**: After power pellet, ghosts flee randomly

## Strategy Tips
1. Learn ghost personalities and predict movements
2. Save power pellets for dangerous situations or ghost chains
3. Use tunnels to escape when surrounded
4. Clear one side of maze before the other for easier escape routes
5. Memorize safe patterns for each level
6. Eat ghosts in succession after power pellets for maximum points

## Level Progression
After completing a level (eating all dots), Pac-Man advances to next level:
- Ghosts become faster
- Power pellet duration decreases
- Chase/scatter timing changes
- Level 21+ (split-screen bug): Game becomes nearly unbeatable

## Technical Information
- Developer: Namco
- Designer: Toru Iwatani
- Release Date: May 22, 1980
- Display: Raster graphics, 224 × 288 pixels
