# Quadrotor Racing Simulation — Technical Reference

This document describes the full physics, control, estimation, and reinforcement-learning stack of the quadrotor racing simulator. Every equation is traced to a specific source file and line range. The intended audience is robotics, controls, and aerospace engineers.

---

## Table of Contents

1. [Notation & Coordinate Frames](#1-notation--coordinate-frames)
2. [Quadrotor Rigid-Body Dynamics](#2-quadrotor-rigid-body-dynamics)
3. [Actuator & Disturbance Models](#3-actuator--disturbance-models)
4. [SE(3) Geometric Controller](#4-se3-geometric-controller)
5. [Sensor Simulation](#5-sensor-simulation)
6. [State Estimation (Error-State EKF)](#6-state-estimation-error-state-ekf)
7. [Gate-Plane Geometry](#7-gate-plane-geometry)
8. [Gate Crossing Logic](#8-gate-crossing-logic)
9. [Reinforcement Learning Interface](#9-reinforcement-learning-interface)
10. [Reward Structure](#10-reward-structure)
11. [System Architecture Summary](#11-system-architecture-summary)

---

## 1. Notation & Coordinate Frames

### Frames

| Frame | Axes | Notes |
|-------|------|-------|
| **World** $\mathcal{W}$ | Z-up, right-handed | Inertial frame. Gravity $\mathbf{g} = [0, 0, -g]^T$, $g = 9.80665$ m/s$^2$ |
| **Body** $\mathcal{B}$ | Z-up through thrust axis | Fixed to the airframe. Thrust acts along body $+z$ |

The rotation matrix $R$ maps body-frame vectors to world-frame vectors:

$$v_{\mathcal{W}} = R\, v_{\mathcal{B}}$$

(`math3d.py:72-73`)

### State Vector

The full state is $\mathbf{x} = (p,\; v,\; q,\; \omega)$ stored in the `State` dataclass (`types.py:16-49`):

| Symbol | Field | Shape | Description |
|--------|-------|-------|-------------|
| $p$ | `p` | $(3,)$ | Position in $\mathcal{W}$ [m] |
| $v$ | `v` | $(3,)$ | Velocity in $\mathcal{W}$ [m/s] |
| $q$ | `q` | $(4,)$ | Attitude quaternion $[w,x,y,z]$ |
| $\omega$ | `w_body` | $(3,)$ | Angular velocity in $\mathcal{B}$ [rad/s] |

### Quaternion Convention

Scalar-first Hamilton convention: $q = [w, x, y, z]$, identity $q_I = [1, 0, 0, 0]$.

The Hamilton product $q_1 \otimes q_2$ represents "first rotate by $q_2$, then by $q_1$" (`math3d.py:29-50`):

$$q_1 \otimes q_2 = \begin{bmatrix} w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\ w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\ w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\ w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2 \end{bmatrix}$$

### Rotation Matrix from Quaternion

Given unit quaternion $q = [w, x, y, z]$ (`math3d.py:87-91`):

$$R = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \end{bmatrix}$$

### Utility Operators

The **hat** map $(\cdot)^\wedge : \mathbb{R}^3 \to \mathfrak{so}(3)$ and its inverse **vee** $(\cdot)^\vee$ (`math3d.py:161-190`):

$$v^\wedge = \begin{bmatrix} 0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0 \end{bmatrix}, \qquad v^\wedge u = v \times u$$

### Control Input

The control vector $u = (T, \tau)$ is held by the `Control` dataclass (`types.py:52-68`):

| Symbol | Field | Units |
|--------|-------|-------|
| $T$ | `thrust_N` | N (scalar, along body $+z$) |
| $\tau$ | `moments_Nm` | N$\cdot$m (3-vector, body-frame torques) |

### Trajectory Setpoint

`TrajPoint` (`types.py:71-99`): $(p_d, v_d, a_d, \psi_d, \dot\psi_d)$.

---

## 2. Quadrotor Rigid-Body Dynamics

Source: `dynamics.py`

### Translational Dynamics

(`dynamics.py:86-100`)

$$m\,\ddot{p} = R \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} + F_{\mathrm{ext}} + m\,\mathbf{g}$$

In code, with optional linear drag $F_{\mathrm{drag}} = -k_d\, v$:

$$\dot{v} = \mathbf{g} + \frac{1}{m}\left(R\begin{bmatrix}0\\0\\T\end{bmatrix} + F_{\mathrm{ext}}\right) - k_d\, v$$

where $\mathbf{g} = [0, 0, -g]^T$.

### Rotational Dynamics (Euler's Equation)

(`dynamics.py:106-111`)

$$J\,\dot{\omega} = \tau + \tau_{\mathrm{ext}} - \omega \times (J\,\omega)$$

Solved as:

$$\dot{\omega} = J^{-1}\bigl(\tau + \tau_{\mathrm{ext}} - \omega \times (J\,\omega)\bigr)$$

### Quaternion Kinematics

(`dynamics.py:102-104`)

$$\dot{q} = \frac{1}{2}\,\Omega(\omega)\, q$$

where the $4 \times 4$ omega matrix (`dynamics.py:16-34`) is:

$$\Omega(\omega) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix}$$

Note: $\Omega$ is skew-symmetric ($\Omega = -\Omega^T$), which preserves the quaternion norm under exact integration.

### RK4 Integration

(`dynamics.py:156-214`)

The state is packed into a single vector $\mathbf{x} \in \mathbb{R}^{13}$:

$$\mathbf{x} = \bigl[\underbrace{p_x, p_y, p_z}_{0{:}3},\; \underbrace{v_x, v_y, v_z}_{3{:}6},\; \underbrace{q_w, q_x, q_y, q_z}_{6{:}10},\; \underbrace{\omega_x, \omega_y, \omega_z}_{10{:}13}\bigr]$$

Standard RK4 with zero-order hold on control and external forces:

$$\begin{aligned}
k_1 &= f(\mathbf{x}) \\
k_2 &= f\!\left(\mathbf{x} + \tfrac{\Delta t}{2}\, k_1\right) \\
k_3 &= f\!\left(\mathbf{x} + \tfrac{\Delta t}{2}\, k_2\right) \\
k_4 &= f\!\left(\mathbf{x} + \Delta t\, k_3\right) \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}$$

After each step, the quaternion is renormalised to unit length (`dynamics.py:212`):

$$q \leftarrow \frac{q}{\|q\|}$$

### Default Physical Parameters

(`params.py:51-62`)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Mass | $m$ | $0.5$ kg |
| Inertia | $J$ | $\mathrm{diag}(0.0023,\; 0.0023,\; 0.004)$ kg$\cdot$m$^2$ |
| Gravity | $g$ | $9.80665$ m/s$^2$ |
| Thrust range | $[T_{\min}, T_{\max}]$ | $[0, 15]$ N |
| Max torque | $\tau_{\max}$ | $[0.1, 0.1, 0.05]$ N$\cdot$m |
| Hover thrust | $mg$ | $4.903$ N |

---

## 3. Actuator & Disturbance Models

### First-Order Actuator Lag

Source: `motor_model.py`

**Continuous-time model** (`motor_model.py:14-16`):

$$\dot{T}_{\mathrm{act}} = \frac{T_{\mathrm{cmd}} - T_{\mathrm{act}}}{\tau_T}$$

**Discrete-time exact ZOH** (`motor_model.py:206-208`):

$$T_{\mathrm{act}}[k+1] = T_{\mathrm{act}}[k] + \alpha\,(T_{\mathrm{cmd}} - T_{\mathrm{act}}[k])$$

where

$$\alpha = 1 - e^{-\Delta t / \tau_T}$$

The same model applies independently to each moment channel with time constant $\tau_\tau$.

### Rate-Limited Pipeline

Per channel, executed each timestep (`motor_model.py:180-219`):

1. **First-order lag** (exact ZOH): compute raw $\Delta_{\mathrm{lag}} = \alpha\,(u_{\mathrm{cmd}} - u_{\mathrm{act}})$
2. **Slew-rate clamp**: $\Delta = \mathrm{clip}(\Delta_{\mathrm{lag}},\; -\dot{u}_{\max}\Delta t,\; +\dot{u}_{\max}\Delta t)$
3. **Integrate**: $u_{\mathrm{act}} \leftarrow u_{\mathrm{act}} + \Delta$
4. **Hard saturation**: $u_{\mathrm{act}} \leftarrow \mathrm{clip}(u_{\mathrm{act}},\; u_{\min},\; u_{\max})$

### Default Actuator Parameters

(`motor_model.py:59-74`)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Thrust time constant | $\tau_T$ | 20 ms |
| Moment time constant | $\tau_\tau$ | 15 ms |
| Thrust slew limit | $\dot{T}_{\max}$ | 200 N/s |
| Moment slew limit | $\dot{\tau}_{\max}$ | $[5, 5, 2.5]$ N$\cdot$m/s |
| Thrust hard limits | $[T_{\min}, T_{\max}]$ | $[0, 15]$ N |
| Moment hard limits | $\pm\tau_{\max}$ | $[0.1, 0.1, 0.05]$ N$\cdot$m |

### Wind & Gust Model

Source: `disturbances.py`

**Constant wind** (`disturbances.py:61-63`): default $v_{\mathrm{wind}} = [0.5, 0.2, 0.0]^T$ m/s.

**Ornstein-Uhlenbeck gust process** (`disturbances.py:146-158`):

Exact discretisation:

$$x_{k+1} = \alpha\, x_k + \sigma_d\, \mathcal{N}(0, I)$$

where

$$\alpha = e^{-\Delta t / \tau_g}, \qquad \sigma_d = \sigma_g \sqrt{\frac{\tau_g}{2}\,(1 - \alpha^2)}$$

The total effective wind velocity is $v_{\mathrm{air}} = v_{\mathrm{wind}} + x$.

### Drag Force

(`disturbances.py:165-169`)

$$F_{\mathrm{drag}} = -k_d\,(v - v_{\mathrm{air}})$$

where $v$ is the vehicle velocity in $\mathcal{W}$ and $v_{\mathrm{air}}$ includes gusts.

### Random Torque Disturbance

(`disturbances.py:172-173`)

$$\tau_{\mathrm{ext}} \sim \mathcal{N}(0,\; \sigma_\tau^2\, I)$$

Applied in the body frame each timestep.

### Default Disturbance Parameters

(`disturbances.py:61-73`)

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Mean wind | $v_{\mathrm{wind}}$ | $[0.5, 0.2, 0.0]$ m/s |
| Gust driving noise | $\sigma_g$ | 0.3 m/s |
| Gust correlation time | $\tau_g$ | 1.0 s |
| Drag coefficient | $k_d$ | 0.15 N$\cdot$s/m |
| Torque disturbance std | $\sigma_\tau$ | 0.0005 N$\cdot$m |

---

## 4. SE(3) Geometric Controller

Source: `controller_se3.py`

Reference: Lee, T., Leok, M., & McClamroch, N. H. (2010). *Geometric Tracking Control of a Quadrotor UAV on SE(3)*.

### Position & Velocity Error

(`controller_se3.py:183-184`)

$$e_p = p - p_d, \qquad e_v = v - v_d$$

### Commanded Acceleration

(`controller_se3.py:190-195`)

$$a_{\mathrm{cmd}} = a_d - K_p\, e_p - K_d\, e_v + g\,\mathbf{e}_3$$

where $\mathbf{e}_3 = [0, 0, 1]^T$ and the gains $K_p$, $K_d$ are diagonal:

| Gain | Symbol | Default |
|------|--------|---------|
| Position P | $K_p$ | $\mathrm{diag}(6, 6, 8)$ |
| Position D | $K_d$ | $\mathrm{diag}(4, 4, 5)$ |

### Desired Rotation Matrix

(`controller_se3.py:39-84`)

The desired body frame $R_d = [\mathbf{b}_{1d}\;\; \mathbf{b}_{2d}\;\; \mathbf{b}_{3d}]$ is constructed from $a_{\mathrm{cmd}}$ and the desired yaw $\psi_d$:

1. Desired thrust axis (body $z$): $\mathbf{b}_{3d} = \dfrac{a_{\mathrm{cmd}}}{\|a_{\mathrm{cmd}}\|}$

2. Desired heading vector: $\mathbf{b}_{1c} = [\cos\psi_d,\; \sin\psi_d,\; 0]^T$

3. Intermediate $y$-axis: $\mathbf{b}_{2d} = \dfrac{\mathbf{b}_{3d} \times \mathbf{b}_{1c}}{\|\mathbf{b}_{3d} \times \mathbf{b}_{1c}\|}$

4. Complete the frame: $\mathbf{b}_{1d} = \mathbf{b}_{2d} \times \mathbf{b}_{3d}$

A degenerate case ($\mathbf{b}_{3d} \parallel \mathbf{b}_{1c}$) is handled by substituting $\mathbf{b}_{1c}$ with $[-\sin\psi_d,\; \cos\psi_d,\; 0]^T$ (`controller_se3.py:70-74`).

### Thrust Computation

(`controller_se3.py:200-201`)

$$T = m\, a_{\mathrm{cmd}} \cdot (R\,\mathbf{e}_3)$$

This projects the desired force onto the current body $z$-axis.

### Attitude Error

(`controller_se3.py:87-109`)

$$e_R = \frac{1}{2}\left(R_d^T R - R^T R_d\right)^\vee$$

### Angular Rate Error

(`controller_se3.py:112-139`)

$$e_\omega = \omega - R^T R_d\,\omega_d$$

where $\omega_d$ is the desired body angular velocity (transformed from the yaw-rate command).

### Moment Command

(`controller_se3.py:225-234`)

$$\tau = -K_r\, e_R - K_\omega\, e_\omega + \omega \times (J\,\omega)$$

The gyroscopic feedforward term $\omega \times (J\,\omega)$ compensates for Coriolis coupling.

| Gain | Symbol | Default |
|------|--------|---------|
| Attitude P | $K_r$ | $\mathrm{diag}(0.1, 0.1, 0.05)$ |
| Attitude D | $K_\omega$ | $\mathrm{diag}(0.02, 0.02, 0.01)$ |

### Saturation

(`controller_se3.py:241-244`)

Element-wise clipping:

$$T_{\mathrm{sat}} = \mathrm{clip}(T,\; T_{\min},\; T_{\max}), \qquad \tau_{\mathrm{sat}} = \mathrm{clip}(\tau,\; -\tau_{\max},\; \tau_{\max})$$

---

## 5. Sensor Simulation

Source: `sensors.py`

All measurements are generated from the **truth** state with additive noise and bias. Deterministic via a seeded `numpy.Generator`.

### Gyroscope (every sim step)

(`sensors.py:170-177`)

$$\omega_{\mathrm{meas}} = \omega_{\mathrm{true}} + b_g + n_g, \qquad n_g \sim \mathcal{N}(0,\; \sigma_g^2\, I)$$

### Accelerometer (every sim step)

(`sensors.py:179-193`)

The accelerometer measures **specific force** in the body frame:

$$a_{\mathrm{meas}} = R^T\!\left(\dot{v} + g\,\mathbf{e}_3\right) + b_a + n_a, \qquad n_a \sim \mathcal{N}(0,\; \sigma_a^2\, I)$$

At hover ($\dot{v} = 0$): $a_{\mathrm{meas}} \approx [0, 0, g]^T$ (correct for a Z-up body frame).

### Bias Random Walk

(`sensors.py:153-162`)

Both gyroscope and accelerometer biases evolve as random walks:

$$b_{k+1} = b_k + \sigma_{\mathrm{rw}}\,\sqrt{\Delta t}\;\mathcal{N}(0, I)$$

### Altimeter

(`sensors.py:198-206`)

$$z_{\mathrm{meas}} = z_{\mathrm{true}} + n_z, \qquad n_z \sim \mathcal{N}(0, \sigma_z^2)$$

Sampled at **50 Hz** (20 ms period). Returns `None` between samples.

### Position Fix

(`sensors.py:211-216`)

$$p_{\mathrm{meas}} = p_{\mathrm{true}} + n_p, \qquad n_p \sim \mathcal{N}(0, \sigma_p^2\, I)$$

Sampled at **20 Hz** (50 ms period). Returns `None` between samples.

### Noise Parameter Summary

(`sensors.py:55-69`)

| Sensor | Parameter | Symbol | Default |
|--------|-----------|--------|---------|
| Gyroscope | White noise std | $\sigma_g$ | 0.01 rad/s |
| Gyroscope | Bias RW driving std | $\sigma_{b_g}$ | 0.0001 rad/s/$\sqrt{\text{s}}$ |
| Accelerometer | White noise std | $\sigma_a$ | 0.1 m/s$^2$ |
| Accelerometer | Bias RW driving std | $\sigma_{b_a}$ | 0.001 m/s$^2$/$\sqrt{\text{s}}$ |
| Altimeter | White noise std | $\sigma_z$ | 0.05 m |
| Altimeter | Update rate | | 50 Hz |
| Position fix | White noise std | $\sigma_p$ | 0.02 m |
| Position fix | Update rate | | 20 Hz |

---

## 6. State Estimation (Error-State EKF)

Source: `estimator_ekf.py`

### State Decomposition

**Nominal state** (stored): $(p, v, q, b_g, b_a)$

**Error state** (15-D): $\delta\mathbf{x} = (\delta p,\; \delta v,\; \delta\theta,\; \delta b_g,\; \delta b_a) \in \mathbb{R}^{15}$

(`estimator_ekf.py:33-41`)

| Index | Slice | Symbol | Dimension |
|-------|-------|--------|-----------|
| 0:3 | `_P` | $\delta p$ | 3 |
| 3:6 | `_V` | $\delta v$ | 3 |
| 6:9 | `_TH` | $\delta\theta$ | 3 |
| 9:12 | `_BG` | $\delta b_g$ | 3 |
| 12:15 | `_BA` | $\delta b_a$ | 3 |

The attitude error is a 3-vector $\delta\theta$ (not a quaternion), giving a minimal 15-D error state despite the 4-component quaternion in the nominal state. This is the key advantage of the error-state formulation.

### Prediction Step

(`estimator_ekf.py:150-215`)

**Bias-corrected IMU**:

$$\hat\omega = \omega_{\mathrm{gyro}} - b_g, \qquad \hat{a} = a_{\mathrm{accel}} - b_a$$

**Nominal state propagation**:

$$q \leftarrow q \otimes \begin{bmatrix}1 \\ \tfrac{1}{2}\hat\omega\,\Delta t\end{bmatrix}$$

$$a_{\mathcal{W}} = R\,\hat{a} + \begin{bmatrix}0 \\ 0 \\ -g\end{bmatrix}$$

$$p \leftarrow p + v\,\Delta t + \tfrac{1}{2}\, a_{\mathcal{W}}\,\Delta t^2$$

$$v \leftarrow v + a_{\mathcal{W}}\,\Delta t$$

Biases are held constant during prediction.

**Continuous-time error-state Jacobian** $F$ (15$\times$15) (`estimator_ekf.py:191-196`):

$$F = \begin{bmatrix}
0_3 & I_3 & 0_3 & 0_3 & 0_3 \\
0_3 & 0_3 & -R[\hat{a}]_\times & 0_3 & -R \\
0_3 & 0_3 & -[\hat\omega]_\times & -I_3 & 0_3 \\
0_3 & 0_3 & 0_3 & 0_3 & 0_3 \\
0_3 & 0_3 & 0_3 & 0_3 & 0_3
\end{bmatrix}$$

where $[\cdot]_\times$ denotes the skew-symmetric (hat) matrix.

Row-by-row interpretation:
- $\delta\dot{p} = \delta v$
- $\delta\dot{v} = -R[\hat{a}]_\times \delta\theta - R\,\delta b_a$
- $\delta\dot\theta = -[\hat\omega]_\times \delta\theta - \delta b_g$
- $\delta\dot{b}_g = 0$ (random walk handled via $Q$)
- $\delta\dot{b}_a = 0$

**First-order discrete transition** (`estimator_ekf.py:199`):

$$\Phi = I_{15} + F\,\Delta t$$

**Discrete process noise** $Q_d$ (`estimator_ekf.py:202-206`):

$$Q_d = \mathrm{diag}\!\left(0_3,\; Q_a\,\Delta t\, I_3,\; Q_g\,\Delta t\, I_3,\; Q_{b_g}\,\Delta t\, I_3,\; Q_{b_a}\,\Delta t\, I_3\right)$$

**Covariance propagation** (`estimator_ekf.py:208`):

$$P \leftarrow \Phi\, P\, \Phi^T + Q_d$$

Symmetry is enforced after propagation: $P \leftarrow \frac{1}{2}(P + P^T)$.

### Measurement Updates

All updates use a **generic EKF update** with the Joseph form (`estimator_ekf.py:230-274`):

$$\begin{aligned}
y &= z - z_{\mathrm{pred}} & &\text{(innovation)} \\
S &= H\, P\, H^T + R & &\text{(innovation covariance)} \\
K &= P\, H^T\, S^{-1} & &\text{(Kalman gain)} \\
\delta\mathbf{x} &= K\, y & &\text{(error-state correction)}
\end{aligned}$$

**Joseph-form covariance update** (`estimator_ekf.py:254-255`):

$$P \leftarrow (I - KH)\, P\, (I - KH)^T + K\, R\, K^T$$

**Error-state injection** (`estimator_ekf.py:258-268`):

- Position: $p \leftarrow p + \delta p$
- Velocity: $v \leftarrow v + \delta v$
- **Attitude** (multiplicative): $q \leftarrow q \otimes \begin{bmatrix}1 \\ \tfrac{1}{2}\delta\theta\end{bmatrix}$
- Biases: $b_g \leftarrow b_g + \delta b_g$, $b_a \leftarrow b_a + \delta b_a$

#### Altimeter Update

(`estimator_ekf.py:281-298`)

$$H_{\mathrm{alt}} = \begin{bmatrix} 0 & 0 & 1 & 0 & \cdots & 0 \end{bmatrix}_{1 \times 15}$$

Observes only $\delta p_z$. Measurement noise: $R = \sigma_z^2$.

#### Position Fix Update

(`estimator_ekf.py:301-317`)

$$H_{\mathrm{pos}} = \begin{bmatrix} I_3 & 0_{3\times 12} \end{bmatrix}_{3 \times 15}$$

Observes $\delta p$. Measurement noise: $R = \sigma_p^2\, I_3$.

### Angular Velocity Estimate

(`estimator_ekf.py:324-349`)

Angular velocity is **not** part of the EKF error state. The controller receives the bias-corrected gyro measurement directly:

$$\hat\omega = \omega_{\mathrm{gyro}} - \hat{b}_g$$

### Default EKF Tuning Parameters

(`estimator_ekf.py:66-84`)

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| $P_0$ position | $P_{0,p}$ | 0.01 | m$^2$ |
| $P_0$ velocity | $P_{0,v}$ | 0.01 | (m/s)$^2$ |
| $P_0$ attitude | $P_{0,\theta}$ | 0.01 | rad$^2$ |
| $P_0$ gyro bias | $P_{0,b_g}$ | $10^{-6}$ | (rad/s)$^2$ |
| $P_0$ accel bias | $P_{0,b_a}$ | $10^{-4}$ | (m/s$^2$)$^2$ |
| Process noise, accel | $Q_a$ | 0.01 | (m/s$^2$)$^2\cdot$s |
| Process noise, gyro | $Q_g$ | $10^{-4}$ | rad$^2$/s |
| Process noise, $b_g$ | $Q_{b_g}$ | $10^{-8}$ | (rad/s)$^2\cdot$s |
| Process noise, $b_a$ | $Q_{b_a}$ | $10^{-6}$ | (m/s$^2$)$^2\cdot$s |

---

## 7. Gate-Plane Geometry

Source: `envs/gates.py`

### Gate Definition

A gate is a circular aperture in 3-D space defined by four quantities (`gates.py:53-75`):

| Field | Symbol | Description |
|-------|--------|-------------|
| `center_w` | $c$ | Gate centre position in $\mathcal{W}$ [m] |
| `normal_w` | $\hat{n}$ | Unit normal pointing in the **direction of travel** |
| `radius_m` | $r$ | Radius of the circular opening [m] |
| `half_thickness_m` | $h$ | Half-thickness of the detection slab [m] |

Defaults: $r = 0.5$ m, $h = 0.2$ m.

### Signed Distance

(`gates.py:83-89`)

$$d(p) = \hat{n} \cdot (p - c)$$

- $d < 0$: drone is **behind** the gate (approaching)
- $d > 0$: drone has **passed through** the gate

### Lateral Projection

To check whether the drone passes *through* the gate opening (not around it):

1. Project the drone position onto the gate plane:

$$p_\perp = p - d\,\hat{n}$$

2. Compute lateral distance from gate centre:

$$\ell = \|p_\perp - c\|$$

3. Crossing is valid only if $\ell \leq r$.

### Normal Construction from Waypoints

(`gates.py:105-153`)

The gate normal is aligned with the local path tangent at waypoint $i$:

- Both neighbours available: $\hat{n} = \mathrm{normalize}(w_{i+1} - w_{i-1})$
- Only predecessor: $\hat{n} = \mathrm{normalize}(w_i - w_{i-1})$
- Only successor: $\hat{n} = \mathrm{normalize}(w_{i+1} - w_i)$

For closed tracks, wrapping indices are used (`gates.py:187-188`).

---

## 8. Gate Crossing Logic

Source: `envs/gate_track.py`

### Hysteresis State Machine

(`gate_track.py:69-84`)

`GateTrack` maintains two hysteresis flags for the current gate:

| Flag | Set when | Meaning |
|------|----------|---------|
| `_was_behind` | $d < -h$ observed | Drone has been solidly behind the gate |
| `_was_ahead` | $d > +h$ observed | Drone has been solidly past the gate |

These prevent spurious crossing detections when the drone hovers near the gate plane.

### Correct Crossing

(`gate_track.py:177-189`)

A **correct crossing** is registered when **all** of:

1. `_was_behind` is `True` (drone was previously behind the gate)
2. $d_{\mathrm{new}} > h$ (drone is now solidly past the gate)
3. Lateral distance $\ell \leq r$ (drone passed through the opening)

On success: gate index advances, hysteresis flags reset, and if wrapping past the last gate, `laps_done` increments.

### Wrong-Direction Crossing

(`gate_track.py:196-200`)

Detected when:

1. `_was_ahead` is `True` (drone was solidly past the gate)
2. $d_{\mathrm{new}} < -h$ (drone is now solidly behind)
3. `_was_behind` was `False` before this step (avoids double-trigger after correct approach)

### Lateral Miss

(`gate_track.py:188-191`)

The gate plane was crossed ($d_{\mathrm{new}} > h$ with `_was_behind`), but $\ell > r$. The `_was_behind` flag is preserved so the drone can re-approach.

### Gate Advance & Lap Counting

(`gate_track.py:203-211`)

On correct crossing:

```
current_gate_idx += 1
if current_gate_idx >= n_gates:
    current_gate_idx = 0
    laps_done += 1
```

Hysteresis flags are re-initialised for the new current gate based on the drone's current signed distance to that gate.

---

## 9. Reinforcement Learning Interface

Source: `envs/quad_racing_env.py`

### Gymnasium Spaces

(`quad_racing_env.py:142-176, 290-296`)

**Observation space**: `Box(-200, 200, shape=(15,), dtype=float32)`

| Index | Content | Source |
|-------|---------|--------|
| 0:3 | Relative position to current gate/WP | $c - p$ or $w_i - p$ |
| 3:6 | Velocity (world frame) | $v$ |
| 6:9 | Body $z$-axis in world frame | $R\,\mathbf{e}_3$ |
| 9:12 | Body angular rates | $\omega$ |
| 12:15 | Gate normal (gate mode) or relative pos to next WP | $\hat{n}$ or $w_{i+1} - p$ |

**Action space**: `Box(-1, 1, shape=(4,), dtype=float32)`

### Action Mapping

(`quad_racing_env.py:394-395`)

$$\Delta a_{\mathcal{W}} = \mathtt{action}[0{:}3] \times a_{\mathrm{res,max}}, \qquad \Delta\dot\psi = \mathtt{action}[3] \times \dot\psi_{\max}$$

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Max residual accel | $a_{\mathrm{res,max}}$ | 5.0 m/s$^2$ |
| Max residual yaw rate | $\dot\psi_{\max}$ | 2.0 rad/s |

### Residual Control Architecture

(`quad_racing_env.py:397-410`)

A **baseline PD controller** (`_baseline_setpoint`, `quad_racing_env.py:539-575`) generates a desired acceleration $a_{d,\mathrm{base}}$ and yaw $\psi_d$ pointing toward the current gate/waypoint. The RL policy adds residuals:

$$a_d = a_{d,\mathrm{base}} + \Delta a_{\mathcal{W}}, \qquad \dot\psi_d = \dot\psi_{\mathrm{base}} + \Delta\dot\psi$$

This combined setpoint is passed to the full SE(3) controller, which computes thrust and moments.

**Why residual control**: The baseline PD keeps the drone stable and roughly on course, so the policy only needs to learn small corrections. This dramatically simplifies exploration compared to learning raw motor commands.

### Baseline PD Parameters

(`quad_racing_env.py:109-112`)

| Parameter | Default |
|-----------|---------|
| $K_{p,\mathrm{outer}}$ | 2.0 |
| $K_{d,\mathrm{outer}}$ | 1.5 |
| $v_{\max}$ | 3.0 m/s |

---

## 10. Reward Structure

Source: `envs/quad_racing_env.py:672-684, 451-455, 462-487`

### Per-Step Reward

$$r = k_{\mathrm{prog}} \cdot \Delta d - k_t - k_c\,\|\mathbf{a}\|^2 + R_{\mathrm{gate}} + R_{\mathrm{success}} - R_{\mathrm{crash}}$$

where the terms are:

| Term | Expression | Default Coefficient |
|------|------------|---------------------|
| Progress | $k_{\mathrm{prog}} \cdot \Delta d$ | $k_{\mathrm{prog}} = 1.0$ |
| Time penalty | $-k_t$ | $k_t = 0.01$ |
| Control penalty | $-k_c \cdot (\sum a_i^2)$ | $k_c = 0.002$ |
| Gate crossing bonus | $+R_{\mathrm{gate}}$ | $R_{\mathrm{gate}} = 10.0$ |
| Track completion | $+R_{\mathrm{success}}$ | $R_{\mathrm{success}} = 100.0$ |
| Crash penalty | $-R_{\mathrm{crash}}$ | $R_{\mathrm{crash}} = 100.0$ |

### Progress Metric

**Gate mode** (`quad_racing_env.py:429-431`):

$$\Delta d = d_{\mathrm{new}} - d_{\mathrm{prev}}$$

where $d$ is the signed distance to the current gate plane. Positive $\Delta d$ means the drone moved *through* the gate.

**Waypoint mode** (`quad_racing_env.py:443`): distance reduction to the current waypoint.

### Termination Conditions

(`quad_racing_env.py:458-491`)

| Condition | Type | Effect |
|-----------|------|--------|
| NaN / Inf in state | Crash | $-R_{\mathrm{crash}}$ |
| $\|p\| > 100$ m | Crash | $-R_{\mathrm{crash}}$ |
| Tilt $> 80°$ | Crash | $-R_{\mathrm{crash}}$ |
| All laps completed | Success | $+R_{\mathrm{success}}$ |
| Wrong-direction crossing (opt.) | Terminated | $-R_{\mathrm{crash}}$ |
| Gate miss (opt.) | Terminated | $-R_{\mathrm{crash}}$ |
| Step count $\geq 2000$ | Truncated | (none) |

---

## 11. System Architecture Summary

### Per-Step Data Flow

```
                                 RL Policy
                                    │
                        action ∈ [-1,1]^4
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │  Residual Mapping    │
                        │  Δa, Δψ̇ scaling     │
                        └────────┬────────────┘
                                 │
                 ┌───────────────┼──────────────────┐
                 │               ▼                   │
                 │    ┌───────────────────┐          │
                 │    │  Baseline PD       │          │
                 │    │  (waypoint/gate)   │          │
                 │    └────────┬──────────┘          │
                 │             │                      │
                 │    a_d = a_base + Δa               │
                 │    ψ̇_d = ψ̇_base + Δψ̇             │
                 │             │                      │
                 │             ▼                      │
                 │    ┌───────────────────┐          │
                 │    │  SE(3) Controller  │◄── estimated state (EKF)
                 │    │  (Lee et al.)      │    or truth state
                 │    └────────┬──────────┘          │
                 │             │                      │
                 │        T_cmd, τ_cmd                │
                 │             │                      │
                 │             ▼                      │
                 │    ┌───────────────────┐          │
                 │    │  Actuator Model    │          │
                 │    │  (1st-order lag +  │          │
                 │    │   rate limit +     │          │
                 │    │   saturation)      │          │
                 │    └────────┬──────────┘          │
                 │             │                      │
                 │        T_applied, τ_applied        │
                 │             │                      │
                 │             ▼                      │
   Wind/Gust ──►│    ┌───────────────────┐          │
   (O-U proc.)  │    │  RK4 Dynamics      │          │
   Drag    ────►│    │  (rigid body)      │          │
   Torque  ────►│    └────────┬──────────┘          │
                 │             │                      │
                 │        truth state                 │
                 │        ┌────┴──────┐              │
                 │        │           │              │
                 │        ▼           ▼              │
                 │    ┌────────┐  ┌──────────────┐  │
                 │    │ Sensors│  │ Gate Crossing │  │
                 │    │ (IMU,  │  │ Detection     │  │
                 │    │  alt,  │  │ (truth pos)   │  │
                 │    │  pos)  │  └──────┬───────┘  │
                 │    └───┬────┘         │          │
                 │        │              │          │
                 │        ▼              ▼          │
                 │    ┌────────┐    ┌─────────┐    │
                 │    │  EKF   │    │ Reward  │    │
                 │    │ (15-D  │    │ Compute │    │
                 │    │ error  │    └────┬────┘    │
                 │    │ state) │         │         │
                 │    └───┬────┘         │         │
                 │        │              │         │
                 │   est. state     reward, done   │
                 │        │              │         │
                 └────────┼──────────────┼─────────┘
                          │              │
                          ▼              ▼
                    observation      (r, terminated,
                    (15-D)           truncated, info)
```

### Separation of Concerns

| Layer | Responsibility | Key Files |
|-------|---------------|-----------|
| **Dynamics** | Rigid-body physics, RK4 integration | `dynamics.py`, `math3d.py` |
| **Actuators** | Motor lag, rate limiting, saturation | `motor_model.py` |
| **Disturbances** | Wind, gusts (O-U), drag, torque noise | `disturbances.py` |
| **Control** | SE(3) geometric tracking | `controller_se3.py` |
| **Sensing** | IMU, altimeter, position fix models | `sensors.py` |
| **Estimation** | Error-state EKF (15-D) | `estimator_ekf.py` |
| **Task** | Gate geometry, crossing detection, track management | `gates.py`, `gate_track.py` |
| **Learning** | Gymnasium env, obs/action/reward, residual control | `quad_racing_env.py` |

### Truth vs. Estimated State

| Consumer | Uses |
|----------|------|
| SE(3) controller | **Estimated** state (when EKF enabled) or truth |
| Observation vector | **Estimated** state (when EKF enabled) or truth |
| Gate crossing detection | **Truth** position (always) |
| Reward computation | **Truth** position (always) |
| Crash detection | **Truth** state (always) |

This design means the policy sees noisy observations (sim-to-real realism), but the reward signal is clean (no noisy credit assignment).

### Timing

| Clock | Rate | Notes |
|-------|------|-------|
| Physics / dynamics | 200 Hz ($\Delta t = 0.005$ s) | RK4 step, actuator, disturbances |
| Control (env step) | 20 Hz (decimation = 10) | SE(3) setpoint held for 10 sim steps |
| IMU (gyro + accel) | 200 Hz | Sampled every sim step |
| Altimeter | 50 Hz | Rate-limited within sensor model |
| Position fix | 20 Hz | Rate-limited within sensor model |
| RL policy | 20 Hz | One action per env step |

---

## Source File Index

| File | Primary Content |
|------|----------------|
| `src/quad/types.py` | `State`, `Control`, `TrajPoint`, `SimLog` dataclasses |
| `src/quad/params.py` | All physical parameters, gains, limits |
| `src/quad/math3d.py` | `quat_to_R`, `quat_mul`, `hat`, `vee`, `R_to_quat` |
| `src/quad/dynamics.py` | `state_derivative`, `step_rk4`, `omega_matrix` |
| `src/quad/controller_se3.py` | SE(3) controller, desired rotation, attitude error |
| `src/quad/motor_model.py` | First-order actuator lag, rate limiting |
| `src/quad/disturbances.py` | Wind, O-U gust process, drag, torque disturbance |
| `src/quad/sensors.py` | IMU, altimeter, position fix noise models |
| `src/quad/estimator_ekf.py` | Error-state EKF: predict, update, state extraction |
| `src/quad/envs/gates.py` | `Gate` dataclass, `waypoints_to_gates` |
| `src/quad/envs/gate_track.py` | `GateTrack` state machine, crossing detection |
| `src/quad/envs/quad_racing_env.py` | Gymnasium env, obs/action/reward, sim loop |
