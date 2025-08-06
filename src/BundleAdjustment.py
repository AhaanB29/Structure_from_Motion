import cv2
import numpy as np

def bundle_adjustment(points_3d, src_pts, dst_pts, K1, K2, R, t):
    """ Simple bundle adjustment to refine camera pose and 3D points. """

    # Ensure shapes and types
    object_points = points_3d.T.astype(np.float32).reshape(-1, 3)       # (N, 3)
    image_points = dst_pts.reshape(-1, 2).astype(np.float32)           # (N, 2)
    camera_matrix = K2.astype(np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

    # Convert R, t to rvec, tvec

    if R is not None and t is not None:
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float32)
    else:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

    # Ensure correct shape


    # Check input validity
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        print("Not enough points to run solvePnP")
        return None, None, None

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("solvePnP failed to find a solution.")
        return None, None, None

    # Convert rvec to rotation matrix
    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec

    
    # Triangulate points again using refined pose
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R_refined, t_refined.T))
    points_4d_hom = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)
    points_3d_refined = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T

    return points_3d_refined, R_refined, t_refined
