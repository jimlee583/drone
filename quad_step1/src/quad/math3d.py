"""
3D math utilities for quaternions and rotations.

Quaternion convention: [w, x, y, z] (scalar-first, Hamilton convention).
Rotation convention: R rotates vectors from body to world frame.
"""

import numpy as np
from numpy.typing import NDArray


def quat_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize a quaternion to unit length.

    Args:
        q: Quaternion [w, x, y, z], shape (4,)

    Returns:
        Normalized quaternion, shape (4,)
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        # Return identity quaternion if input is near-zero
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_mul(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Multiply two quaternions (Hamilton product).

    q1 * q2 represents: first rotate by q2, then by q1.

    Args:
        q1: First quaternion [w, x, y, z], shape (4,)
        q2: Second quaternion [w, x, y, z], shape (4,)

    Returns:
        Product quaternion, shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute quaternion conjugate.

    For unit quaternions, conjugate equals inverse.

    Args:
        q: Quaternion [w, x, y, z], shape (4,)

    Returns:
        Conjugate quaternion, shape (4,)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_R(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion to rotation matrix.

    The rotation matrix R rotates vectors from body to world frame:
        v_world = R @ v_body

    Args:
        q: Unit quaternion [w, x, y, z], shape (4,)

    Returns:
        Rotation matrix, shape (3, 3)
    """
    # Ensure normalized
    q = quat_normalize(q)
    w, x, y, z = q

    # Rotation matrix from quaternion
    # Using direct formula for efficiency
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return R


def R_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert rotation matrix to quaternion.

    Uses Shepperd's method for numerical stability.

    Args:
        R: Rotation matrix, shape (3, 3)

    Returns:
        Unit quaternion [w, x, y, z], shape (4,)
    """
    # Shepperd's method - choose the largest diagonal element to avoid
    # numerical issues when trace is small
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Ensure w > 0 for consistency (choose canonical form)
    if w < 0:
        q = -q
    return quat_normalize(q)


def quat_rotate_vec(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotate a vector by a quaternion.

    Equivalent to R(q) @ v but using quaternion math directly.

    Args:
        q: Unit quaternion [w, x, y, z], shape (4,)
        v: Vector to rotate, shape (3,)

    Returns:
        Rotated vector, shape (3,)
    """
    # For efficiency, use the rotation matrix
    R = quat_to_R(q)
    return R @ v


def hat(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the skew-symmetric (hat) matrix of a 3D vector.

    hat(v) @ u = v × u (cross product)

    Args:
        v: 3D vector, shape (3,)

    Returns:
        Skew-symmetric matrix, shape (3, 3)
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def vee(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Extract the vector from a skew-symmetric matrix (inverse of hat).

    Args:
        M: Skew-symmetric matrix, shape (3, 3)

    Returns:
        3D vector, shape (3,)
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def wrap_angle_pi(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX convention (yaw-pitch-roll).
    Only for visualization/plotting purposes.

    Args:
        q: Unit quaternion [w, x, y, z], shape (4,)

    Returns:
        Euler angles [roll, pitch, yaw] in radians, shape (3,)
    """
    w, x, y, z = quat_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Handle gimbal lock
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def safe_normalize(v: NDArray[np.float64], fallback: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Safely normalize a vector, returning fallback if near-zero.

    Args:
        v: Vector to normalize, shape (3,)
        fallback: Fallback unit vector if v is near-zero, shape (3,)

    Returns:
        Normalized vector or fallback, shape (3,)
    """
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return fallback
    return v / norm


# ============================================================================
# Unit tests (run with: python -m quad.math3d)
# ============================================================================

if __name__ == "__main__":
    print("Running math3d unit tests...")

    # Test 1: Quaternion normalization
    q = np.array([1.0, 1.0, 1.0, 1.0])
    q_norm = quat_normalize(q)
    assert np.abs(np.linalg.norm(q_norm) - 1.0) < 1e-10, "Quaternion normalization failed"
    print("  [PASS] quat_normalize")

    # Test 2: Identity quaternion -> identity rotation
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_R(q_id)
    assert np.allclose(R_id, np.eye(3)), "Identity quaternion should give identity matrix"
    print("  [PASS] quat_to_R identity")

    # Test 3: Rotation matrix is orthonormal
    q_test = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    R_test = quat_to_R(q_test)
    assert np.allclose(R_test @ R_test.T, np.eye(3)), "R should be orthonormal (R @ R.T = I)"
    assert np.allclose(np.linalg.det(R_test), 1.0), "det(R) should be 1"
    print("  [PASS] quat_to_R orthonormality")

    # Test 4: Quaternion multiplication
    # 90 degree rotation about z, then 90 about y should give specific result
    q_z90 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # 90 deg about z
    q_y90 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])  # 90 deg about y
    q_combined = quat_mul(q_y90, q_z90)  # First z, then y
    R_combined = quat_to_R(q_combined)
    R_expected = quat_to_R(q_y90) @ quat_to_R(q_z90)
    assert np.allclose(R_combined, R_expected), "Quaternion multiplication failed"
    print("  [PASS] quat_mul")

    # Test 5: Quaternion conjugate
    q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
    q_conj = quat_conj(q)
    q_prod = quat_mul(q, q_conj)
    assert np.allclose(q_prod, [1, 0, 0, 0]), "q * conj(q) should be identity"
    print("  [PASS] quat_conj")

    # Test 6: hat/vee roundtrip
    v = np.array([1.0, 2.0, 3.0])
    v_hat = hat(v)
    v_recovered = vee(v_hat)
    assert np.allclose(v, v_recovered), "hat/vee roundtrip failed"
    print("  [PASS] hat/vee roundtrip")

    # Test 7: hat is skew-symmetric
    v = np.array([1.0, 2.0, 3.0])
    v_hat = hat(v)
    assert np.allclose(v_hat, -v_hat.T), "hat(v) should be skew-symmetric"
    print("  [PASS] hat skew-symmetry")

    # Test 8: hat implements cross product
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    cross_direct = np.cross(u, v)
    cross_hat = hat(u) @ v
    assert np.allclose(cross_direct, cross_hat), "hat(u) @ v should equal u × v"
    print("  [PASS] hat cross product")

    # Test 9: R_to_quat is inverse of quat_to_R
    q_orig = quat_normalize(np.array([0.7, 0.3, 0.5, 0.4]))
    R = quat_to_R(q_orig)
    q_back = R_to_quat(R)
    R_back = quat_to_R(q_back)
    assert np.allclose(R, R_back), "R_to_quat should be inverse of quat_to_R"
    print("  [PASS] R_to_quat / quat_to_R roundtrip")

    # Test 10: wrap_angle_pi
    assert np.abs(wrap_angle_pi(0.0)) < 1e-10, "wrap_angle_pi(0) should be 0"
    # Note: pi and -pi are equivalent; modulo typically gives -pi for pi input
    assert np.abs(np.abs(wrap_angle_pi(np.pi)) - np.pi) < 1e-10, "wrap_angle_pi(pi) should be +/-pi"
    assert np.abs(wrap_angle_pi(2*np.pi)) < 1e-10, "wrap_angle_pi(2pi) should be 0"
    assert np.abs(np.abs(wrap_angle_pi(-np.pi)) - np.pi) < 1e-10, "wrap_angle_pi(-pi) should be +/-pi"
    assert np.abs(np.abs(wrap_angle_pi(3*np.pi)) - np.pi) < 1e-10, "wrap_angle_pi(3pi) should be +/-pi"
    print("  [PASS] wrap_angle_pi")

    # Test 11: Euler conversion sanity check
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    euler = quat_to_euler(q_id)
    assert np.allclose(euler, [0, 0, 0]), "Identity quaternion should give zero Euler angles"
    print("  [PASS] quat_to_euler identity")

    print("\nAll math3d tests passed!")
