import numpy as np
import cv2
from scipy.optimize import least_squares

def _ensure_pts(pts):
    pts = np.asarray(pts)
    if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T
    if pts.ndim == 1 and pts.size == 2:
        pts = pts.reshape(1, 2)
    return pts.astype(np.float64)

def _project_points(X, rvec, tvec, K):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    X_cam = (R @ X.T).T + tvec.reshape(1, 3)
    z_safe = np.where(np.abs(X_cam[:, 2:3]) < 1e-8, 1e-8, X_cam[:, 2:3])
    xy = X_cam[:, :2] / z_safe
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * xy[:, 0] + cx
    v = fy * xy[:, 1] + cy
    return np.stack([u, v], axis=1)

def bundle_adjustment(points_3d, src_pts, dst_pts, K1, K2, R=None, t=None,
                      robust_loss='huber', loss_f_scale=1.0, max_nfev=200, verbose=0):
    """
    Faster local BA: refine camera2 pose (R, t) and 3D points.
    """

    X = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    N = X.shape[0]
    pts1 = _ensure_pts(src_pts)
    pts2 = _ensure_pts(dst_pts)
    if pts1.shape[0] != N or pts2.shape[0] != N:
        raise ValueError("3D and 2D points must match in number")

    # Initial pose
    if R is None or t is None:
        rvec0 = np.zeros(3)
        tvec0 = np.zeros(3)
    else:
        rvec0, _ = cv2.Rodrigues(np.asarray(R).reshape(3, 3))
        rvec0 = rvec0.ravel()
        tvec0 = np.asarray(t).reshape(3)

    K1 = np.asarray(K1, dtype=np.float64)
    K2 = np.asarray(K2, dtype=np.float64)

    def pack_params(rvec, tvec, X):
        return np.hstack([rvec.ravel(), tvec.ravel(), X.ravel()])

    def unpack_params(p):
        rvec = p[0:3]
        tvec = p[3:6]
        pts = p[6:].reshape(-1, 3)
        return rvec, tvec, pts

    def residuals(params):
        rvec, tvec, pts = unpack_params(params)
        proj1 = _project_points(pts, np.zeros(3), np.zeros(3), K1)
        proj2 = _project_points(pts, rvec, tvec, K2)
        res1 = (proj1 - pts1).ravel()
        res2 = (proj2 - pts2).ravel()
        return np.hstack([res1, res2])

    # Sparse Jacobian pattern: each point affects only its own rows and the 6 pose params
    m = 4 * N  # residuals
    n = 6 + 3 * N  # parameters
    J_pattern = np.zeros((m, n), dtype=bool)
    for i in range(N):
        # First 2 residuals (image1)
        J_pattern[2 * i, 0:6] = False
        J_pattern[2 * i + 1, 0:6] = False
        J_pattern[2 * i:2 * i + 2, 6 + 3 * i: 6 + 3 * i + 3] = True
        # Next 2 residuals (image2)
        J_pattern[2 * N + 2 * i: 2 * N + 2 * i + 2, 0:6] = True
        J_pattern[2 * N + 2 * i: 2 * N + 2 * i + 2, 6 + 3 * i: 6 + 3 * i + 3] = True

    x0 = pack_params(rvec0, tvec0, X)

    res = least_squares(
        residuals, x0, jac_sparsity=J_pattern,
        loss=robust_loss, f_scale=loss_f_scale,
        max_nfev=max_nfev, verbose=2 if verbose else 0
    )

    rvec_opt, tvec_opt, pts_opt = unpack_params(res.x)
    R_refined, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
    return pts_opt, R_refined, tvec_opt.reshape(3, 1), res
