"""
Main simulation loop.

Orchestrates the interaction between dynamics, controller,
and trajectory for a complete simulation run.

The simulation pipeline per timestep is:
    1. Controller  →  commanded control (thrust + moments)
    2. Actuator model  →  applied control (with lag, rate limit, saturation)
    3. Disturbance model  →  external forces (wind drag) and torques (gusts)
    4. RK4 dynamics step  →  next state

Both the actuator model and disturbances are toggleable via params.
"""

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from quad.types import State, Control, TrajPoint, SimLog
from quad.params import Params, default_params
from quad.dynamics import step_rk4
from quad.controller_se3 import se3_controller
from quad.log import allocate_log, record_step
from quad.motor_model import ActuatorState, step_actuator
from quad.disturbances import DisturbanceState, compute_disturbance_forces


def run_sim(
    params: Params,
    traj_fn: Callable[[float], TrajPoint],
    t_final: float,
    dt: float = 0.002,
    x0: Optional[State] = None,
    verbose: bool = False,
) -> SimLog:
    """
    Run a complete quadrotor simulation.

    Simulates the quadrotor from initial state, tracking the given
    trajectory using the SE(3) controller and RK4 integration.

    The loop now includes:
    - First-order actuator dynamics (motor lag + rate limiting + saturation)
    - Environmental disturbances (wind drag + gusts + torque noise)

    Both are enabled by default but can be disabled via ``params.actuator``
    and ``params.wind``.

    Args:
        params: Quadrotor and controller parameters
        traj_fn: Trajectory function returning TrajPoint for time t
        t_final: Simulation end time [s]
        dt: Integration timestep [s] (default: 0.002 = 500 Hz)
        x0: Initial state (default: origin, zero velocity, identity attitude)
        verbose: Print progress updates

    Returns:
        SimLog containing complete simulation history
    """
    # Number of timesteps (include t=0)
    n_steps = int(np.ceil(t_final / dt)) + 1

    # Initialize state
    if x0 is None:
        state = State.zeros()
    else:
        state = x0.copy()

    # Initialize actuator state at hover so there is no initial transient
    # when the quad is already near hover equilibrium.
    act_state = ActuatorState.from_hover(params.hover_thrust)

    # Initialize disturbance state with the configured seed.
    dist_state = DisturbanceState.create(seed=params.wind.seed)

    # Pre-allocate log
    log = allocate_log(n_steps)

    # Simulation loop
    t = 0.0
    step = 0
    
    if verbose:
        print(f"Starting simulation: t_final={t_final}s, dt={dt*1000:.1f}ms, steps={n_steps}")
        if params.actuator.enabled:
            print(f"  Actuator model ON  (tau_T={params.actuator.tau_T*1e3:.0f}ms)")
        if params.wind.enabled:
            print(f"  Wind model ON  (mean={params.wind.wind_vel}, gust_std={params.wind.gust_std})")

    while t <= t_final and step < n_steps:
        # Get desired trajectory at current time
        traj = traj_fn(t)

        # 1) Controller  →  commanded control
        cmd_control, ctrl_state = se3_controller(state, traj, params)

        # 2) Actuator model  →  applied (lagged/saturated) control
        act_state, applied_control = step_actuator(
            act_state, cmd_control, params.actuator, dt,
        )

        # 3) Disturbance model  →  external wrench
        F_ext, tau_ext, dist_state = compute_disturbance_forces(
            state.v, dist_state, params.wind, dt,
        )

        # Log current state (record the *applied* control, not the command,
        # so plots reflect what the dynamics actually saw).
        record_step(
            log,
            t=t,
            state=state,
            control=applied_control,
            cmd_control=cmd_control,
            traj=traj,
            e_pos=ctrl_state.e_pos,
            e_vel=ctrl_state.e_vel,
            e_att=ctrl_state.e_att,
            e_rate=ctrl_state.e_rate,
        )

        # 4) Integrate dynamics with applied control + external wrench
        state = step_rk4(
            state, applied_control, params, dt,
            F_ext=F_ext, tau_ext=tau_ext,
        )

        # Advance time
        t += dt
        step += 1

        # Progress update
        if verbose and step % 1000 == 0:
            pos_err = np.linalg.norm(ctrl_state.e_pos)
            print(f"  t={t:.2f}s, pos_err={pos_err*1000:.1f}mm")

    # Trim log to actual number of recorded steps
    log = log.trim()

    if verbose:
        print(f"Simulation complete: {step} steps recorded")

    return log


def run_hover_test(
    params: Optional[Params] = None,
    hover_height: float = 1.0,
    t_final: float = 5.0,
    dt: float = 0.002,
) -> SimLog:
    """
    Run a simple hover test.

    Starts at ground level and tracks hover at specified height.

    Args:
        params: Parameters (default: default_params())
        hover_height: Target hover altitude [m]
        t_final: Simulation time [s]
        dt: Timestep [s]

    Returns:
        SimLog
    """
    from quad.trajectory import hover

    if params is None:
        params = default_params()

    traj_fn = hover(z=hover_height)
    return run_sim(params, traj_fn, t_final, dt)


def run_step_test(
    params: Optional[Params] = None,
    target: NDArray[np.float64] = None,
    t_final: float = 8.0,
    dt: float = 0.002,
) -> SimLog:
    """
    Run a step response test.

    Starts at origin and steps to target position.

    Args:
        params: Parameters (default: default_params())
        target: Target position (default: [1, 1, 1])
        t_final: Simulation time [s]
        dt: Timestep [s]

    Returns:
        SimLog
    """
    from quad.trajectory import step_to

    if params is None:
        params = default_params()
    if target is None:
        target = np.array([1.0, 1.0, 1.0])

    traj_fn = step_to(target_p=target, t_step=1.0)
    return run_sim(params, traj_fn, t_final, dt)


def run_circle_test(
    params: Optional[Params] = None,
    radius: float = 1.0,
    speed: float = 0.5,
    t_final: float = 20.0,
    dt: float = 0.002,
) -> SimLog:
    """
    Run a circular trajectory test.

    Args:
        params: Parameters (default: default_params())
        radius: Circle radius [m]
        speed: Tangential speed [m/s]
        t_final: Simulation time [s]
        dt: Timestep [s]

    Returns:
        SimLog
    """
    from quad.trajectory import circle

    if params is None:
        params = default_params()

    traj_fn = circle(radius=radius, speed=speed, z=1.0)
    
    # Start at the initial point of the circle
    x0 = State.zeros()
    x0.p = np.array([radius, 0.0, 1.0])
    
    return run_sim(params, traj_fn, t_final, dt, x0=x0)


def run_figure8_test(
    params: Optional[Params] = None,
    a: float = 1.0,
    b: float = 0.5,
    speed: float = 0.5,
    t_final: float = 30.0,
    dt: float = 0.002,
) -> SimLog:
    """
    Run a figure-8 trajectory test.

    Args:
        params: Parameters (default: default_params())
        a: X amplitude [m]
        b: Y amplitude [m]
        speed: Average speed [m/s]
        t_final: Simulation time [s]
        dt: Timestep [s]

    Returns:
        SimLog
    """
    from quad.trajectory import figure8

    if params is None:
        params = default_params()

    traj_fn = figure8(a=a, b=b, speed=speed, z=1.0)
    
    # Start at the initial point of the figure-8 (origin in xy, at altitude)
    x0 = State.zeros()
    x0.p = np.array([0.0, 0.0, 1.0])
    
    return run_sim(params, traj_fn, t_final, dt, x0=x0)


if __name__ == "__main__":
    """Quick simulation test."""
    print("Running simulation test...")

    from quad.log import print_statistics

    # Run a quick hover test
    params = default_params()
    log = run_hover_test(params, hover_height=1.0, t_final=2.0)

    print_statistics(log, "Hover Test")

    # Check that we reached the target.
    # Tolerance is 50 mm to account for steady-state offset from wind
    # disturbances (the SE(3) controller has no integral term).
    final_pos = log.p[-1]
    assert np.abs(final_pos[2] - 1.0) < 0.05, f"Should be near z=1.0, got {final_pos[2]}"
    print(f"  [PASS] Hover reached target altitude (z={final_pos[2]:.4f} m)")

    print("\nSimulation test passed!")
