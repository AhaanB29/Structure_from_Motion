# Requires: pip install scipy
import numpy as np
import cv2
from scipy.optimize import least_squares

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R

def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.flatten()

def pack_params(cameras, points3d, cam_idx_map):
    """
    Pack parameters into a 1D vector for optimization.
    We only pack parameters for cameras that are not fixed (cam['fixed']==False).
    cam_idx_map: dict mapping camera id -> index in 'cameras' list
    Return: x0, meta where meta describes slices.
    """
    cam_params = []
    cam_slices = {}
    for i, cam in enumerate(cameras):
        if cam.get("fixed", False):
            cam_slices[i] = None
            continue
        # rvec (3,) and tvec (3,)
        r = np.asarray(cam["rvec"]).reshape(3)
        t = np.asarray(cam["tvec"]).reshape(3)
        start = len(cam_params)
        cam_params.extend(r.tolist())
        cam_params.extend(t.tolist())
        cam_slices[i] = (start, start + 6)

    pts_start = len(cam_params)
    pts_len = points3d.size  # flattened length = 3*M
    x0 = np.hstack([np.asarray(cam_params, dtype=np.float64), points3d.flatten().astype(np.float64)])
    meta = {
        "cam_slices": cam_slices,
        "pts_start": pts_start,
        "n_points": points3d.shape[0]
    }
    return x0, meta

def unpack_params(x, cameras, meta):
    cam_slices = meta["cam_slices"]
    pts_start = meta["pts_start"]
    n_points = meta["n_points"]
    # Update cameras
    for i, cam in enumerate(cameras):
        slice_info = cam_slices[i]
        if slice_info is None:
            continue
        s, e = slice_info
        rvec = x[s:s+3]
        tvec = x[s+3:s+6]
        cam["rvec"] = rvec
        cam["tvec"] = tvec
    # Update points
    pts_flat = x[pts_start:pts_start + 3 * n_points]
    points3d = pts_flat.reshape((n_points, 3))
    return cameras, points3d

def reprojection_residuals(x, cameras, points3d_init, observations, meta, robust_loss_scale=1.0):
    """
    Compute reprojection residuals for all observations.
    observations: list of (pt_idx, cam_idx, xy)
    Returns residual vector of length 2 * len(observations)
    """
    # Unpack x into current param values (but avoid copying points repeatedly)
    cam_slices = meta["cam_slices"]
    pts_start = meta["pts_start"]
    n_points = meta["n_points"]

    # Extract points
    pts_flat = x[pts_start:pts_start + 3 * n_points]
    points = pts_flat.reshape((n_points, 3))

    residuals = []
    for (pt_idx, cam_idx, xy) in observations:
        # xy: measured pixel coords (2,)
        slice_info = cam_slices[cam_idx]
        if slice_info is None:
            # camera is fixed: take rvec/tvec from cameras list
            rvec = np.asarray(cameras[cam_idx]["rvec"], dtype=np.float64).reshape(3)
            tvec = np.asarray(cameras[cam_idx]["tvec"], dtype=np.float64).reshape(3)
        else:
            s, e = slice_info
            rvec = x[s:s+3]
            tvec = x[s+3:s+6]

        # Intrinsics: expect camera dict has 'K' as 3x3
        K = np.asarray(cameras[cam_idx]["K"], dtype=np.float64)

        # Project point
        R = rodrigues_to_R(rvec)
        pw = points[pt_idx].reshape(3,1)
        pc = R.dot(pw) + tvec.reshape(3,1)   # (3,1)
        if pc[2,0] <= 1e-8:
            # behind camera - produce large residual to discourage this configuration
            residuals.append(1000.0)  # x
            residuals.append(1000.0)  # y
            continue

        proj = (K @ pc).flatten()
        proj = proj[:2] / proj[2]

        res = proj - np.asarray(xy, dtype=np.float64).reshape(2)
        residuals.append(res[0])
        residuals.append(res[1])

    return np.asarray(residuals, dtype=np.float64)

def bundle_adjustment_local(cameras, points3d, observations, max_nfev=200, verbose=2):
    """
    Jointly optimize camera rvec/tvec for non-fixed cameras and 3D points.

    cameras: list where index == camera index, each is dict {
        "id": int,
        "K": 3x3,
        "rvec": (3,),
        "tvec": (3,),
        "fixed": bool (optional, default False)
    }
    points3d: (M,3) ndarray initial
    observations: list of (pt_idx (0..M-1), cam_idx (0..C-1), xy (2,))
        Each observation is a measurement of a 3D point in a camera image.
    Returns: cameras_updated, points3d_refined
    """
    # create mapping cam id -> index in list if needed (we assume cameras are in list order)
    cam_idx_map = {cam["id"]: i for i, cam in enumerate(cameras)}

    x0, meta = pack_params(cameras, points3d, cam_idx_map)

    # residual wrapper uses symmetric Huber-like loss via least_squares' loss param
    def fun(x):
        return reprojection_residuals(x, cameras, points3d, observations, meta)

    # Use robust loss (Huber) to be more like COLMAP's robust BA
    res = least_squares(fun, x0, jac='2-point', verbose=verbose, x_scale='jac', ftol=1e-8, xtol=1e-8,
                        gtol=1e-8, max_nfev=max_nfev, loss='huber', f_scale=1.0)

    cameras_opt, points_opt = unpack_params(res.x, cameras, meta)
    return cameras_opt, points_opt
