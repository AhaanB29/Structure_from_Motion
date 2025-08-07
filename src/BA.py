import cv2
import numpy as np

def bundle_adjustment(points_3d, src_pts, dst_pts, K1, K2, R, t):
    """ Enhanced bundle adjustment to refine camera pose and 3D points. """
    
    # Ensure shapes and types
    object_points = points_3d.T.astype(np.float32).reshape(-1, 3)  # (N, 3)
    image_points = dst_pts.reshape(-1, 2).astype(np.float32)  # (N, 2)
    camera_matrix = K2.astype(np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion
    
    # Fix shape issue: ensure t is (3, 1)
    if t.shape == (1, 3):
        t = t.T  # Convert (1, 3) to (3, 1)
    
    # Convert R, t to rvec, tvec
    if R is not None and t is not None:
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float32).reshape(3, 1)  # Ensure (3, 1) shape
    else:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
    
    # Check input validity - increased threshold
    if object_points.shape[0] < 6 or image_points.shape[0] < 6:
        print("Not enough points to run solvePnP")
        return points_3d.T, R, t
    
    # Check for degenerate configurations
    if np.linalg.matrix_rank(object_points) < 3:
        print("Degenerate 3D point configuration")
        return points_3d.T, R, t
    
    try:
        # Solve PnP with RANSAC for robustness
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, dist_coeffs,
            rvec, tvec, useExtrinsicGuess=True, 
            iterationsCount=1000, reprojectionError=2.0,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 6:
            print(f"solvePnP failed or insufficient inliers: {len(inliers) if inliers is not None else 0}")
            return points_3d.T, R, t
        
        print(f"Bundle adjustment: {len(inliers)}/{len(object_points)} inliers")
        
        # Convert rvec to rotation matrix
        R_refined, _ = cv2.Rodrigues(rvec)
        t_refined = tvec.reshape(3, 1)  # Ensure (3, 1) shape
        
        # Refine with iterative method using inliers only
        if len(inliers) >= 6:
            inlier_indices = inliers.flatten()
            object_points_inliers = object_points[inlier_indices]
            image_points_inliers = image_points[inlier_indices]
            
            success_refine, rvec_refine, tvec_refine = cv2.solvePnP(
                object_points_inliers, image_points_inliers, 
                camera_matrix, dist_coeffs,
                rvec, tvec, useExtrinsicGuess=True, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success_refine:
                R_refined, _ = cv2.Rodrigues(rvec_refine)
                t_refined = tvec_refine.reshape(3, 1)
        
        # Fix triangulation - ensure correct point formats
        # src_pts and dst_pts should be (N, 2) arrays
        if src_pts.shape[1] != 2:
            src_pts_formatted = src_pts.T  # Convert to (N, 2)
        else:
            src_pts_formatted = src_pts
            
        if dst_pts.shape[1] != 2:
            dst_pts_formatted = dst_pts.T  # Convert to (N, 2)
        else:
            dst_pts_formatted = dst_pts
        
        # Triangulate points again using refined pose
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R_refined, t_refined))
        
        # Ensure points are in correct format for triangulation (2, N)
        points_4d_hom = cv2.triangulatePoints(
            P1, P2, 
            src_pts_formatted.T.astype(np.float32),  # Convert to (2, N)
            dst_pts_formatted.T.astype(np.float32)   # Convert to (2, N)
        )
        
        # Convert to 3D points
        points_3d_refined = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T
        
        # Validate triangulated points
        depths1 = points_3d_refined @ np.eye(3).T  # For camera 1 (identity)
        depths2 = points_3d_refined @ R_refined.T + t_refined.T  # For camera 2
        
        # Check if points are in front of both cameras
        valid_mask = (depths1[:, 2] > 0.1) & (depths2[:, 2] > 0.1)
        
        if np.sum(valid_mask) < len(points_3d_refined) * 0.5:
            print("Too many points behind cameras, keeping original")
            return points_3d.T, R, t
        
        # Filter points with reasonable depth
        distance_mask = np.linalg.norm(points_3d_refined, axis=1) < 20
        final_mask = valid_mask & distance_mask
        
        if np.sum(final_mask) >= 6:
            points_3d_refined = points_3d_refined[final_mask]
            print(f"Filtered to {len(points_3d_refined)} valid points")
        
        return points_3d_refined, R_refined, t_refined
        
    except Exception as e:
        print(f"Bundle adjustment failed with exception: {e}")
        return points_3d.T, R, t