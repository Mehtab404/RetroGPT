# BREAKOUT - Game Manual

## Game Overview
Breakout is a brick-breaking arcade game developed by Atari and released in 1976. Control a paddle to bounce a ball and break all bricks at the top of the screen.

## Controls
- **Dial/Paddle**: Move paddle left and right along bottom of screen
- Ball automatically launches at start and after each life lost

## Gameplay

### Objective
Break all bricks using the ball without letting it fall past the paddle. Clear all eight rows of bricks to advance to the next level.

### Brick Layout
Eight rows of bricks in two colors:
- **Rows 1-2 (closest to top)**: Yellow bricks - 7 points each
- **Rows 3-4**: Green bricks - 5 points each
- **Rows 5-6**: Orange bricks - 3 points each  
- **Rows 7-8 (closest to paddle)**: Red bricks - 1 point each

Each row contains multiple bricks spanning the width of the playfield.

### Ball Mechanics
- Ball bounces off walls, paddle, and bricks
- Angle of deflection depends on where ball hits paddle
- Ball speeds up slightly after hitting bricks in top rows
- Ball speeds up significantly after breaking through to top
- Only one ball active at a time

### Paddle Control
- Paddle width remains constant throughout game
- Paddle moves horizontally along bottom screen
- Ball deflection angle determined by hit position:
  - Center: Shallow angle
  - Edges: Steep angle
  - Use for aiming ball trajectory

## Scoring System
- Red bricks (bottom rows): 1 point
- Orange bricks: 3 points
- Green bricks: 5 points
- Yellow bricks (top rows): 7 points

## Lives System
- Game starts with 3-5 balls (lives), depending on settings
- Lose a ball when it passes paddle and falls off bottom
- No extra balls awarded
- Game ends when all balls are lost

## Level Progression
After breaking all bricks:
- New level starts with fresh brick layout
- Ball speed remains at current speed (doesn't reset)
- Difficulty increases as ball maintains accumulated speed
- Paddle stays same size

## Strategy Tips

1. **Top Rows Priority**
   - Yellow and green bricks worth more points
   - Breaking through to top creates "tunnel"

2. **Tunnel Strategy**
   - Break vertical path through bricks
   - Ball bounces rapidly in cleared space above
   - Can clear many bricks quickly
   - Dangerous - hard to control when ball returns

3. **Paddle Positioning**
   - Stay centered for better reach
   - Use paddle edges to angle shots deliberately
   - Don't chase ball desperately - predict trajectory

4. **Angle Control**
   - Hit ball with paddle edge for steep angles
   - Aim for gaps in brick rows
   - Create strategic openings

5. **Speed Management**
   - Ball accelerates in later stages
   - Anticipate ball position, don't react
   - Focus on paddle positioning

6. **Brick Priority**
   - Clear bottom rows first for safety (easier to hit)
   - Or risk breaking through for tunnel strategy
   - Avoid single bricks in middle - hard to target

## Advanced Techniques

### Edge Shots
Master hitting ball with extreme paddle edges to create nearly vertical angles, useful for targeting specific bricks.

### Rhythm Timing
In tunnel situations, develop timing rhythm for when ball will return from top, allowing better paddle positioning.

### Controlled Tunneling
Create tunnel at strategic position (not center) to control ball return angle and maintain better control.

## Variants

### Breakout Sequels
- **Super Breakout** (1978): Added features like multiple balls, paddle shrinking, progressive difficulty
- Later variants added power-ups, different brick patterns, multiple levels

## Technical Information
- Developer: Atari, Inc.
- Concept: Nolan Bushnell, Steve Bristow
- Release Date: May 13, 1976
- Display: Black and white raster with color overlay
- Historical Note: Steve Jobs and Steve Wozniak built prototype (Wozniak's design was too complex for production, was redesigned)

## Gameplay Loop
1. Ball launches from paddle
2. Ball bounces off walls and paddle
3. Ball breaks bricks on contact
4. Collect points based on brick color
5. Prevent ball from falling past paddle
6. Clear all bricks to complete level
7. Repeat with increased difficulty

## Trivia
- Breakout inspired Taito's Space Invaders
- Core gameplay influenced entire "brick breaker" genre
- Simplified one-player version of Pong concept
- Wozniak's chip-efficient design techniques later used in Apple II
