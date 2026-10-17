# ASTEROIDS - Game Manual

## Game Overview
Asteroids is a space-themed multidirectional shooter arcade game released by Atari in 1979. Players control a triangular spaceship in an asteroid field, destroying asteroids and flying saucers while avoiding collisions.

## Controls
- **ROTATE LEFT**: Rotate the ship counterclockwise
- **ROTATE RIGHT**: Rotate the ship clockwise  
- **THRUST**: Move the ship forward in the direction it's facing
- **FIRE**: Shoot bullets to destroy asteroids and saucers
- **HYPERSPACE**: Emergency teleport to a random location (risky!)

## Gameplay

### Objective
Destroy all asteroids and enemy saucers while surviving as long as possible to achieve the highest score.

### Asteroids
The playing field contains asteroids of three sizes:
- **Large asteroids**: 20 points when destroyed, break into 2 medium asteroids
- **Medium asteroids**: 50 points when destroyed, break into 2 small asteroids  
- **Small asteroids**: 100 points when destroyed, completely eliminated

### Flying Saucers
Periodically, flying saucers appear and shoot at your ship:
- **Large saucer**: 200 points, appears early, shoots randomly
- **Small saucer**: 1000 points, appears later, aims accurately at player

### Ship Mechanics
- Your ship has momentum - thrust accelerates you, but you keep drifting
- Ship wraps around screen edges (exit left, appear right)
- Collision with any asteroid or saucer destroys your ship
- You start with 3 ships (lives)

### Hyperspace
- Press hyperspace to instantly teleport to random location
- 25% chance ship explodes upon rematerialization
- Use only in emergencies

## Scoring System
- Small asteroid: 100 points
- Medium asteroid: 50 points
- Large asteroid: 20 points
- Large saucer: 200 points
- Small saucer: 1000 points
- Extra ship awarded at 10,000 points

## Strategy Tips
1. Break large asteroids early to prevent screen clutter
2. Keep moving - stationary targets are easy prey for saucers
3. Use ship momentum to navigate efficiently
4. Small saucers are dangerous but worth high points
5. Save hyperspace for true emergencies

## Wave Progression
After clearing all asteroids, a new wave begins with more asteroids. Each wave increases difficulty with more asteroids and earlier saucer appearances.

## Technical Information
- Developer: Atari, Inc.
- Designer: Lyle Rains, Ed Logg
- Release Date: November 1979
- Display: Vector graphics (240 lines)
