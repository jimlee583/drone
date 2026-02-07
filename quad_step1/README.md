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
- **First-order actuator dynamics** — motor lag, rate limiting, and hard saturation
- **Environmental disturbances** — constant wind, stochastic gusts, and aerodynamic drag

## Installation

### Using uv (Recommended)

```bash
cd quad_step1
uv sync
```

Then run:
```bash
uv run python -m quad.main
```

### Alternative: pip

```bash
cd quad_step1
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
python -m quad.main
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
    ├── motor_model.py      # First-order actuator dynamics (Step 2)
    ├── disturbances.py     # Wind and gust disturbance model (Step 2)
    ├── sim.py              # Main simulation loop
    ├── log.py              # Simulation logging utilities
    ├── plots.py            # Visualization functions
    ├── main.py             # Entry point
    ├── scenarios.py        # Evaluation scenario registry & param randomization
    ├── metrics.py          # Evaluation metrics (RMS error, saturation, crash)
    └── evaluate.py         # Monte Carlo evaluation CLI
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

## Actuator & Disturbance Realism (Step 2)

The simulation now includes two optional (enabled-by-default) realism layers
that sit between the controller output and the rigid-body dynamics:

### First-Order Actuator Model (`motor_model.py`)

Real motors cannot change thrust instantaneously.  A first-order lag models
the combined ESC + motor-winding + propeller aerodynamic delay:

    dT/dt = (T_cmd − T_actual) / τ_T

- **Time constants**: τ\_T = 20 ms (thrust), τ\_τ = 15 ms (moments).
- **Rate limiting**: slew rate is clamped so the actuator cannot exceed a
  physically-realizable rate of change.
- **Hard saturation**: thrust and moments are clipped to hardware limits.
- Exact zero-order-hold (ZOH) discretisation is used for numerical stability.

### Environmental Disturbances (`disturbances.py`)

Outdoor flight conditions are modelled as:

- **Constant mean wind** in the world frame (default: mild breeze at 0.5 m/s).
- **Stochastic gusts** via a discrete Ornstein–Uhlenbeck process with
  configurable correlation time and intensity.
- **Linear aerodynamic drag**: F\_wind = −k\_drag · (v − v\_wind).
- **Random torque noise** (small, body-frame) for prop-wash asymmetries.

All disturbance RNG uses a fixed seed (42) for reproducibility.

### Toggling

Both features can be disabled independently for comparison or debugging:

```python
from quad.params import Params
from quad.motor_model import ActuatorParams
from quad.disturbances import WindParams

# Disable actuator lag, keep wind
params = Params(actuator=ActuatorParams(enabled=False))

# Disable wind, keep actuator lag
params = Params(wind=WindParams(enabled=False))

# Disable both (Step-1 behaviour)
params = Params(
    actuator=ActuatorParams(enabled=False),
    wind=WindParams(enabled=False),
)
```

## Monte Carlo Evaluation Suite

A built-in evaluation harness randomizes physical parameters and runs
repeated trials to assess controller robustness.

### Quick Start

```bash
# List available scenarios
python -m quad.evaluate --list-scenarios

# Run 50-trial evaluation on hover
python -m quad.evaluate --scenario hover --trials 50 --seed 42 --verbose

# Run on all scenarios
for s in hover step circle figure8; do
  python -m quad.evaluate --scenario $s --trials 50
done
```

### Scenarios

| Name     | Duration | Trajectory                             |
|----------|----------|----------------------------------------|
| hover    | 10 s     | Static hold at z = 1 m                 |
| step     | 10 s     | Step to [1, 0, 1] at t = 1 s          |
| circle   | 30 s     | Radius 3 m, speed 2 m/s, z = 1 m      |
| figure8  | 40 s     | a = 2, b = 1, speed 1.5 m/s, z = 1 m  |

### Metrics

- **rms_pos_err / max_pos_err** — position tracking error [m]
- **rms_vel_err** — velocity tracking error [m/s]
- **control_effort** — integrated thrust^2 + ||moments||^2
- **thrust_sat_pct** — % timesteps at thrust limits
- **moment_sat_{roll,pitch,yaw,overall}** — % timesteps at moment limits
- **crashed** — divergence / NaN / quaternion norm check

### Randomization Ranges

| Parameter              | Range              |
|------------------------|--------------------|
| Mass                   | ± 10 %             |
| Inertia (per axis)     | ± 15 %             |
| Motor time constants   | ± 30 %             |
| Wind velocity          | 0 – 3 m/s (random) |
| Gust seed              | random              |

### Output

Results are written to `results/` as CSV (one row per trial) and JSON
(full config + per-trial metrics + aggregate stats).

## State Estimation (Step 3)

An optional error-state EKF can be enabled so that the controller runs off
estimated (noisy) state instead of truth state.

### Sensor Model (`sensors.py`)

- **IMU** (gyro + accelerometer) sampled every sim step with configurable
  white noise and bias random-walk.
- **Barometric altimeter** at 50 Hz (configurable).
- **Position fix** (vision / GPS stand-in) at 20 Hz (configurable).

### Error-State EKF (`estimator_ekf.py`)

- 15-D error state: δp, δv, δθ, δb\_g, δb\_a.
- Prediction from bias-corrected IMU.
- Scalar altimeter update (z) and 3-D position-fix update.
- Quaternion error injection via small-angle approximation.

### Toggling

```python
from quad.params import Params

params = Params(use_estimator=True)   # controller uses EKF estimate
params = Params(use_estimator=False)  # controller uses truth (default)
```

### CLI

```bash
# Demo with estimator
python -m quad.main --trajectory hover --use-estimator

# Evaluation with estimator
python -m quad.evaluate --scenario hover --trials 5 --seed 1 --use-estimator

# Evaluation without estimator (default, unchanged)
python -m quad.evaluate --scenario hover --trials 5 --seed 1
```

### Determinism

Sensor noise is seeded via `SensorParams.seed` (default 0).  During Monte
Carlo evaluation the seed is randomized per trial automatically.

## Future Extensions (Step 4+)

- Gym environment wrapper for RL
- Domain randomization

## References

1. Lee, T., Leok, M., & McClamroch, N. H. (2010). Geometric tracking control of a
   quadrotor UAV on SE(3). CDC 2010.
2. Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control
   for quadrotors. ICRA 2011.
