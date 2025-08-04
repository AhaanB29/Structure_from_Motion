import cv2
import numpy as np

def bundle_adjustment(points_3d, 
                      src_pts, dst_pts, 
                      K1, R1, t1,     # Intrinsics/extrinsics for the first (reference) camera
                      K2, R2, t2):    # Intrinsics/extrinsics for the second (to refine) camera
    """
    Refine the pose (R2, t2) of camera 2 and re-triangulate points between camera 1 and 2.
    - points_3d: 3×N array of existing 3D points in world coords.
    - src_pts:   N×2 array of reprojected 2D points in camera1 (assumed perfect).
    - dst_pts:   N×2 array of measured 2D points in camera2.
    - K1, R1, t1: intrinsics/extrinsics of camera1 (reference).
    - K2, R2, t2: intrinsics/extrinsics of camera2 (initial guess + to refine).
    """

    # Prepare for solvePnP
    obj_pts = points_3d.T.astype(np.float64)      # shape: N×3
    img_pts = dst_pts.astype(np.float64)          # shape: N×2
    cam_mat = K2           # use K2 for camera2
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Initial guess for rvec2, tvec2
    rvec2, _ = cv2.Rodrigues(R2)
    tvec2 = t2.reshape(3, 1).astype(np.float64)

    # Only attempt PnP if enough points
    if obj_pts.shape[0] < 4:
        raise ValueError("Need at least 4 points for PnP.")

    # Refine camera2 pose
    success, rvec2, tvec2 = cv2.solvePnP(
        obj_pts, img_pts, cam_mat, dist_coeffs,
        rvec2, tvec2,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed.")

    # Build refined R2, t2
    R2_refined, _ = cv2.Rodrigues(rvec2)
    t2_refined = tvec2.flatten()

    # Re-triangulate between camera1 and camera2 using their respective Ks
    # Build projection matrices with correct intrinsics:
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K2 @ np.hstack((R2_refined, t2_refined.reshape(3, 1)))

    # Triangulate using the original correspondences (src_pts in cam1, dst_pts in cam2)
    pts1 = src_pts.T  # shape: 2×N
    pts2 = dst_pts.T  # shape: 2×N
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d_refined = (points_4D[:3, :] / points_4D[3, :]).T  # shape: N×3
    #print(pts3d_refined.shape)
    return pts3d_refined, R2_refined, t2_refined
