# SPACE INVADERS - Game Manual

## Game Overview
Space Invaders is a fixed shooter arcade game developed by Taito in 1978. Control a laser cannon, moving horizontally across the bottom of the screen, firing at descending aliens.

## Controls
- **LEFT**: Move laser cannon left
- **RIGHT**: Move laser cannon right
- **FIRE**: Shoot laser at invaders (one bullet on screen at a time)

## Gameplay

### Objective
Eliminate all invaders before they reach the bottom of the screen. Survive waves of increasingly fast invaders to achieve the highest score.

### Invaders Formation
55 invaders arranged in 5 rows of 11:
- **Top Row (Squid)**: 30 points each
- **Middle Rows (Crab)**: 20 points each (2 rows)
- **Bottom Rows (Octopus)**: 10 points each (2 rows)

### Movement Pattern
- Invaders move side-to-side as a group
- When reaching screen edge, they drop down one row and reverse direction
- As invaders are destroyed, remaining ones speed up
- Last remaining invader moves fastest

### Mystery Ship (UFO)
- Appears periodically at top of screen
- Moves horizontally across screen
- Worth 50-300 points (sometimes predictable based on shot timing)
- Does not fire at player
- 300 points if hit after exactly 23 shots

## Shields/Bunkers
Four destructible bunkers protect the player:
- Positioned between player and invaders
- Gradually destroyed by invader fire and player shots
- Provide temporary cover
- Cannot be repaired or rebuilt

## Game Mechanics

### Firing
- Only one player bullet allowed on screen at a time
- Must wait for bullet to hit target or reach top before firing again
- Invaders fire randomly, multiple shots can be active

### Speed Progression
- Invaders speed increases as more are destroyed
- Speed resets at start of new wave
- Iconic "heartbeat" sound speeds up with invaders

### Defeat Conditions
- Player loses life if hit by invader fire
- Player loses life if any invader reaches bottom of screen
- Game over when all lives lost (typically 3 lives)

## Scoring System
- Octopus (bottom rows): 10 points
- Crab (middle rows): 20 points  
- Squid (top row): 30 points
- Mystery Ship: 50, 100, 150, or 300 points
- Extra life typically at 1,500 points

## Wave Progression
After destroying all 55 invaders:
- New wave begins with fresh formation
- Invaders start slightly lower on screen
- Shields regenerate
- Difficulty increases with each wave

## Strategy Tips
1. Shoot through gaps in formations to hit multiple rows
2. Clear bottom rows first - they score fewer points but pose more danger
3. Memorize UFO scoring patterns for 300-point shots
4. Use shields strategically, creating firing lanes
5. Don't shoot shields unnecessarily - preserve them
6. Shoot corners of formation to slow overall movement
7. Keep one invader alive longer to practice and maximize UFO points

## Advanced Techniques

### UFO Scoring Pattern
The 300-point UFO follows a pattern:
- Count shots fired (including misses)
- 300 points awarded on 23rd shot after previous UFO
- Requires precise timing and shot counting

### Rainbow Pattern
Experienced players can manipulate invader positions to create safe zones where invader shots don't reach player position.

## Technical Information
- Developer: Taito
- Designer: Tomohiro Nishikado
- Release Date: 1978
- Display: Black and white raster with color overlay
- Impact: Popularized the shooter genre and video games globally
