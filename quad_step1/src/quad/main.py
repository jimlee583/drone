"""
Main entry point for quadrotor simulation.

Run with: python -m quad.main

Examples:
    python -m quad.main                  # Run all trajectories
    python -m quad.main --trajectory hover
    python -m quad.main --trajectory step
    python -m quad.main --trajectory circle
    python -m quad.main --trajectory figure8
    python -m quad.main --no-plot        # Run without showing plots
"""

import argparse
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from quad.params import Params, default_params
from quad.sim import (
    run_hover_test,
    run_step_test,
    run_circle_test,
    run_figure8_test,
)
from quad.log import print_statistics
from quad.plots import (
    plot_xy_path,
    plot_3d_path,
    plot_pos_time,
    plot_errors,
    plot_controls,
    plot_tilt_angle,
)
from quad.types import SimLog


def run_hover(params: Params, show_plots: bool = True) -> SimLog:
    """Run hover trajectory simulation."""
    print("\n" + "=" * 60)
    print("HOVER TEST")
    print("=" * 60)
    print("Target: Hold position at [0, 0, 1] m")
    print("Starting from: [0, 0, 0] m (ground)")
    
    log = run_hover_test(params, hover_height=1.0, t_final=5.0)
    print_statistics(log, "Hover")
    
    if show_plots:
        plot_pos_time(log, "Hover: Position vs Time")
        plot_errors(log, "Hover: Tracking Errors")
        plot_controls(log, "Hover: Control Inputs")
    
    return log


def run_step(params: Params, show_plots: bool = True) -> SimLog:
    """Run step response simulation."""
    print("\n" + "=" * 60)
    print("STEP RESPONSE TEST")
    print("=" * 60)
    print("Target: Move from [0, 0, 0] to [1, 1, 1] m")
    print("Step occurs at t=1s with smooth transition")
    
    log = run_step_test(
        params,
        target=np.array([1.0, 1.0, 1.0]),
        t_final=8.0,
    )
    print_statistics(log, "Step Response")
    
    if show_plots:
        plot_3d_path(log, "Step: 3D Path")
        plot_pos_time(log, "Step: Position vs Time")
        plot_errors(log, "Step: Tracking Errors")
        plot_controls(log, "Step: Control Inputs")
    
    return log


def run_circle(params: Params, show_plots: bool = True) -> SimLog:
    """Run circular trajectory simulation."""
    print("\n" + "=" * 60)
    print("CIRCLE TRACKING TEST")
    print("=" * 60)
    print("Trajectory: Circle with radius=1m, speed=0.5m/s, z=1m")
    print("Duration: ~2 full laps")
    
    log = run_circle_test(
        params,
        radius=1.0,
        speed=0.5,
        t_final=30.0,  # About 2.4 laps
    )
    print_statistics(log, "Circle Tracking")
    
    if show_plots:
        plot_xy_path(log, "Circle: XY Path")
        plot_3d_path(log, "Circle: 3D Path")
        plot_errors(log, "Circle: Tracking Errors")
        plot_controls(log, "Circle: Control Inputs")
        plot_tilt_angle(log, "Circle: Tilt Angle")
    
    return log


def run_figure8(params: Params, show_plots: bool = True) -> SimLog:
    """Run figure-8 trajectory simulation."""
    print("\n" + "=" * 60)
    print("FIGURE-8 TRACKING TEST")
    print("=" * 60)
    print("Trajectory: Figure-8 with a=1m, b=0.5m, speed=0.5m/s")
    print("Duration: ~2 full laps")
    
    log = run_figure8_test(
        params,
        a=1.0,
        b=0.5,
        speed=0.5,
        t_final=40.0,
    )
    print_statistics(log, "Figure-8 Tracking")
    
    if show_plots:
        plot_xy_path(log, "Figure-8: XY Path")
        plot_3d_path(log, "Figure-8: 3D Path")
        plot_errors(log, "Figure-8: Tracking Errors")
        plot_controls(log, "Figure-8: Control Inputs")
        plot_tilt_angle(log, "Figure-8: Tilt Angle")
    
    return log


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quadrotor Dynamics & SE(3) Control Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quad.main                    # Run all trajectories
  python -m quad.main --trajectory hover # Run only hover test
  python -m quad.main --no-plot          # Run without plots
        """,
    )
    
    parser.add_argument(
        "--trajectory", "-t",
        type=str,
        choices=["hover", "step", "circle", "figure8", "all"],
        default="all",
        help="Which trajectory to run (default: all)",
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot display",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 60)
    print("  QUAD STEP 1: Quadrotor Dynamics & Nonlinear SE(3) Control")
    print("=" * 60)
    
    # Initialize parameters
    params = default_params()
    print(f"\nQuadrotor Parameters:")
    print(f"  Mass: {params.m} kg")
    print(f"  Hover thrust: {params.hover_thrust:.2f} N")
    print(f"  Thrust limits: [{params.T_min}, {params.T_max}] N")
    
    show_plots = not args.no_plot
    
    # Run requested trajectory(s)
    if args.trajectory == "all":
        run_hover(params, show_plots=show_plots)
        run_step(params, show_plots=show_plots)
        run_circle(params, show_plots=show_plots)
        run_figure8(params, show_plots=show_plots)
    elif args.trajectory == "hover":
        run_hover(params, show_plots=show_plots)
    elif args.trajectory == "step":
        run_step(params, show_plots=show_plots)
    elif args.trajectory == "circle":
        run_circle(params, show_plots=show_plots)
    elif args.trajectory == "figure8":
        run_figure8(params, show_plots=show_plots)
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    if show_plots:
        print("\nDisplaying plots... Close plot windows to exit.")
        plt.show()
    else:
        print("\nPlots disabled. Use without --no-plot to see visualizations.")


if __name__ == "__main__":
    main()
