import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    xyz = -q[..., :3]
    w = q[..., 3:]
    return torch.cat([xyz, w], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def euler_xyz_to_quat(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles.unbind(dim=-1)
    hx = 0.5 * x
    hy = 0.5 * y
    hz = 0.5 * z

    qx = torch.stack([
        torch.sin(hx),
        torch.zeros_like(hx),
        torch.zeros_like(hx),
        torch.cos(hx),
    ], dim=-1)
    qy = torch.stack([
        torch.zeros_like(hy),
        torch.sin(hy),
        torch.zeros_like(hy),
        torch.cos(hy),
    ], dim=-1)
    qz = torch.stack([
        torch.zeros_like(hz),
        torch.zeros_like(hz),
        torch.sin(hz),
        torch.cos(hz),
    ], dim=-1)

    # Matches scipy Rotation.from_euler("xyz", angles).as_quat() in xyzw order.
    return quat_normalize(quat_mul(quat_mul(qz, qy), qx))


def relative_quat_from_euler_pairs(angles_1: torch.Tensor, angles_2: torch.Tensor) -> torch.Tensor:
    q1 = euler_xyz_to_quat(angles_1)
    q2 = euler_xyz_to_quat(angles_2)
    return quat_normalize(quat_mul(quat_inv(q1), q2))


def quat_relative(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    return quat_normalize(quat_mul(quat_inv(q_from), q_to))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    if t.ndim == q0.ndim - 1:
        t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    linear = quat_normalize((1 - t) * q0 + t * q1)
    w0 = torch.sin((1 - t) * theta) / sin_theta.clamp_min(1e-8)
    w1 = torch.sin(t * theta) / sin_theta.clamp_min(1e-8)
    spherical = quat_normalize(w0 * q0 + w1 * q1)

    use_linear = sin_theta.abs() < 1e-6
    return torch.where(use_linear, linear, spherical)
