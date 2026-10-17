# CENTIPEDE - Game Manual

## Game Overview
Centipede is a fixed shooter arcade game by Atari released in 1981. Players defend against a centipede, spiders, fleas, and scorpions using a small insect-like shooter at the bottom of the screen.

## Controls
- **Trackball/Joystick**: Move shooter in all directions (limited to bottom portion of screen)
- **FIRE**: Rapid-fire weapon to destroy enemies and mushrooms

## Gameplay

### Objective
Destroy the attacking centipede before it reaches the bottom of the screen. Clear waves of increasingly difficult enemies to achieve high scores.

### The Centipede
- Segmented creature consisting of 10-12 segments
- Moves horizontally until hitting mushroom or screen edge, then drops down
- When hit, splits into multiple shorter centipedes
- Each segment destroyed: 10 points
- Head segment: 100 points when centipede fully destroyed

### Centipede Behavior
- Starts at top of screen
- Descends through mushroom field
- Changes direction when hitting mushrooms
- Splits at point of impact when shot
- Multiple centipedes can be active simultaneously
- Reaches player area faster as level progresses

## Enemies

### Spider
- Bounces diagonally across lower screen
- Eats mushrooms in player area
- Awards variable points based on distance when destroyed:
  - Near: 900 points
  - Medium: 600 points  
  - Far: 300 points
- Can destroy player on contact

### Flea
- Drops vertically down screen when few mushrooms remain in player area
- Leaves trail of mushrooms behind
- 200 points when destroyed
- Requires 2 hits to destroy (1 hit in early levels)
- Only appears when mushroom count is low

### Scorpion
- Moves horizontally across mid-screen
- Poisons mushrooms it touches (turn purple/red)
- Poisoned mushrooms make centipede drop straight down
- 1000 points when destroyed
- Does not directly attack player

## Mushrooms
- Fill playfield at start
- Take 4 hits to destroy completely (change color each hit)
- Worth 1 point per hit
- Regenerate partially between waves
- Poisoned mushrooms cause centipede to drop vertically

## Player Shooter
- Moves freely in bottom 1/4 of screen
- Can venture slightly into mushroom field
- Rapid-fire capability
- Starts with 3 lives
- Extra life typically at 10,000 and 20,000 points

## Scoring System
- Mushroom hit: 1 point (4 points to fully destroy)
- Centipede body segment: 10 points
- Centipede head (last segment): 100 points
- Flea: 200 points
- Spider: 300-900 points (distance-dependent)
- Scorpion: 1000 points

## Wave Progression
After destroying entire centipede:
- New wave starts with fresh centipede
- Centipede speed increases
- More aggressive enemy spawning
- Mushroom field partially regenerates
- Faster flea and spider appearances

## Strategy Tips
1. Create "corridors" in mushrooms to guide centipede movement
2. Clear mushrooms near bottom to slow centipede descent
3. Shoot spiders when close for maximum points
4. Prioritize scorpions - prevent mushroom poisoning
5. Destroy centipede heads to prevent splitting
6. Manage mushroom field - don't create too many gaps
7. Stay mobile - don't get cornered by spider
8. Shoot fleas immediately to prevent mushroom spawning

## Advanced Techniques

### Mushroom Management
- Maintain mushroom density to slow centipede
- Create strategic patterns for predictable centipede paths
- Clear vertical lanes for safe zones

### Spider Hunting
Position shooter to maximize close-range spider kills (900 pts) while maintaining safety from centipede.

### Scorpion Priority
Always prioritize scorpions - poisoned mushrooms dramatically increase difficulty by causing centipede to drop straight down.

## Technical Information
- Developer: Atari, Inc.
- Designer: Dona Bailey, Ed Logg
- Release Date: 1981
- Display: Raster graphics, 256 × 231 pixels
- Notable: Co-designed by Dona Bailey, one of first female arcade game designers
