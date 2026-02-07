# Quad Step 1: Quadrotor Dynamics & Nonlinear SE(3) Control

This project implements a quadrotor dynamics simulator with a geometric SE(3) tracking controller.
It is the first step in building a complete quadrotor simulation framework, with future steps
adding reinforcement learning capabilities.

## Features

- **Rigid body quadrotor dynamics** with quaternion attitude representation
- **Geometric SE(3) controller** for position and attitude tracking
- **RK4 integration** with configurable timestep (default 0.002s)
- **Analytic reference trajectories**: hover, step, circle, figure-8
- **Thrust and moment saturation** for realistic actuator limits
- **Comprehensive logging and plotting** for validation

## Installation

### Option 1: Editable Install (Recommended)

```bash
cd quad_step1
pip install -e .
```

Then run:
```bash
python -m quad.main
```

### Option 2: Using PYTHONPATH

```bash
cd quad_step1
PYTHONPATH=src python -m quad.main
```

## Usage

The main script runs different trajectory scenarios:

```bash
# Run all trajectories (default)
python -m quad.main

# Run specific trajectory
python -m quad.main --trajectory hover
python -m quad.main --trajectory step
python -m quad.main --trajectory circle
python -m quad.main --trajectory figure8
```

## Project Structure

```
quad_step1/
├── pyproject.toml          # Project configuration
├── README.md               # This file
└── src/quad/
    ├── __init__.py         # Package init
    ├── types.py            # Dataclasses: State, Control, TrajPoint, SimLog
    ├── math3d.py           # Quaternion and rotation utilities
    ├── params.py           # Quadrotor parameters and gains
    ├── dynamics.py         # Rigid body dynamics with RK4 integration
    ├── controller_se3.py   # Geometric SE(3) tracking controller
    ├── trajectory.py       # Reference trajectory generators
    ├── sim.py              # Main simulation loop
    ├── log.py              # Simulation logging utilities
    ├── plots.py            # Visualization functions
    └── main.py             # Entry point
```

## Technical Details

### Coordinate Frames

- **World frame (W)**: North-East-Down (NED) with Z pointing up
- **Body frame (B)**: X forward, Y right, Z up (out of top of quad)

### State Representation

- Position `p`: 3D position in world frame [m]
- Velocity `v`: 3D velocity in world frame [m/s]
- Quaternion `q`: Attitude as [w, x, y, z] (scalar-first convention)
- Angular velocity `w_body`: Body-frame angular rates [rad/s]

### Controller

The SE(3) geometric controller follows Lee et al. "Geometric Tracking Control of a
Quadrotor UAV on SE(3)" (CDC 2010). It provides:

1. Position tracking via desired acceleration command
2. Attitude tracking via geometric error on SO(3)
3. Feedforward terms for trajectory tracking

### Default Parameters

- Mass: 0.5 kg (typical 250-500g racing quad)
- Inertia: Diagonal [0.0023, 0.0023, 0.004] kg·m²
- Thrust limits: [0, 15] N
- Position gains: Kp = [6, 6, 8], Kd = [4, 4, 5]
- Attitude gains: Kr = [0.1, 0.1, 0.05], Kw = [0.01, 0.01, 0.005]

## Expected Results

### Hover
- Quad should stabilize at z=1.0m within ~1 second
- Position error < 1mm in steady state
- Minimal control effort after stabilization

### Step Response
- Smooth transition to target position
- No overshoot with default gains
- Settling time ~2-3 seconds

### Circle Tracking
- Smooth circular path following
- Small tracking lag due to finite gains
- Position error < 5cm for 1m radius at 0.5 m/s

### Figure-8 Tracking
- Continuous smooth tracking of complex trajectory
- Largest errors at the crossover point
- Demonstrates coupled position/attitude control

## Future Extensions (Step 2+)

- Gym environment wrapper for RL
- Motor dynamics and mixing
- Sensor noise and state estimation
- Wind disturbances
- Domain randomization

## References

1. Lee, T., Leok, M., & McClamroch, N. H. (2010). Geometric tracking control of a
   quadrotor UAV on SE(3). CDC 2010.
2. Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control
   for quadrotors. ICRA 2011.
