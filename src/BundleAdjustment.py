import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def global_bundle_adjustment(cameras, points_3d, observations, visited_ids,
                           robust_loss='huber', loss_f_scale=1.0, 
                           max_nfev=100, verbose=0):
    """
    Global Bundle Adjustment with gauge anchoring - optimizes cameras and 3D points
    
    Args:
        cameras: Dict {img_id: {'R': R, 't': t, 'K': K}}
        points_3d: List of 3D points [[x,y,z,r,g,b], ...]
        observations: List of {camera_id, point_idx, pixel} dicts
        visited_ids: List of camera IDs in order (for anchoring)
    
    Returns:
        optimized_cameras, optimized_points, success
    """
    
    if len(cameras) < 3 or len(points_3d) < 10:
        print(f"[BA] Insufficient data: {len(cameras)} cameras, {len(points_3d)} points")
        return cameras, points_3d, False
    
    print(f"[BA] Starting BA with {len(cameras)} cameras, {len(points_3d)} points, {len(observations)} observations")
    
    # Create camera ID to index mapping
    cam_ids = list(cameras.keys())
    cam_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(cam_ids)}
    
    # Identify anchored cameras (gauge constraints)
    first_cam_id = visited_ids[0] if len(visited_ids) > 0 else cam_ids[0]
    second_cam_id = visited_ids[1] if len(visited_ids) > 1 else cam_ids[1]
    
    # Anchor second camera for early reconstructions to prevent scale drift
    anchor_second = len(visited_ids) < 15
    
    if verbose:
        anchored = [first_cam_id]
        if anchor_second:
            anchored.append(second_cam_id)
        print(f"[BA] Anchoring cameras: {anchored}")
    
    def pack_parameters():
        """Pack camera poses and 3D points into parameter vector"""
        params = []
        
        # Pack camera parameters (6 DOF each: rvec + tvec)
        for cam_id in cam_ids:
            # Skip anchored cameras
            if cam_id == first_cam_id or (anchor_second and cam_id == second_cam_id):
                continue
                
            cam = cameras[cam_id]
            rvec, _ = cv2.Rodrigues(cam['R'])
            params.extend(rvec.flatten())
            params.extend(cam['t'].flatten())
        
        # Pack 3D points (3 DOF each)
        for point in points_3d:
            params.extend(point[:3])  # Only x, y, z (not colors)
            
        return np.array(params, dtype=np.float64)
    
    def unpack_parameters(params):
        """Unpack parameters back to cameras and points"""
        param_idx = 0
        new_cameras = {}
        
        # Unpack camera parameters
        for cam_id in cam_ids:
            if cam_id == first_cam_id or (anchor_second and cam_id == second_cam_id):
                # Keep anchored cameras unchanged
                new_cameras[cam_id] = cameras[cam_id].copy()
            else:
                rvec = params[param_idx:param_idx+3]
                tvec = params[param_idx+3:param_idx+6]
                param_idx += 6
                
                R, _ = cv2.Rodrigues(rvec)
                new_cameras[cam_id] = {
                    'R': R,
                    't': tvec.reshape(3, 1),
                    'K': cameras[cam_id]['K'].copy()
                }
        
        # Unpack 3D points
        new_points = []
        for i, point in enumerate(points_3d):
            point_coords = params[param_idx:param_idx+3]
            param_idx += 3
            
            # Keep original colors
            new_point = [point_coords[0], point_coords[1], point_coords[2], 
                        point[3], point[4], point[5]]
            new_points.append(new_point)
            
        return new_cameras, new_points
    
    def residuals(params):
        """Compute reprojection residuals"""
        # Unpack parameters
        param_idx = 0
        cam_params = {}
        
        # Handle anchored cameras
        for cam_id in cam_ids:
            if cam_id == first_cam_id or (anchor_second and cam_id == second_cam_id):
                # Use original parameters for anchored cameras
                cam = cameras[cam_id]
                rvec, _ = cv2.Rodrigues(cam['R'])
                cam_params[cam_id] = (rvec.flatten(), cam['t'].flatten())
            else:
                rvec = params[param_idx:param_idx+3]
                tvec = params[param_idx+3:param_idx+6]
                param_idx += 6
                cam_params[cam_id] = (rvec, tvec)
        
        # Unpack 3D points
        point_coords = {}
        for i in range(len(points_3d)):
            point_coords[i] = params[param_idx:param_idx+3]
            param_idx += 3
        
        # Compute residuals for each observation
        residuals_list = []
        for obs in observations:
            cam_id = obs['camera_id']
            point_idx = obs['point_idx']
            observed_pixel = obs['pixel']
            
            # Get camera and point parameters
            rvec, tvec = cam_params[cam_id]
            K = cameras[cam_id]['K']
            point_3d = point_coords[point_idx].reshape(1, 3)
            
            # Project to image
            # A try-except block is still good practice for rare numerical errors
            try:
                projected_pixels, _ = cv2.projectPoints(point_3d, rvec, tvec, K, None)
                projected_pixel = projected_pixels[0, 0]
                
                residual = projected_pixel - observed_pixel
                residuals_list.extend(residual)
            except cv2.error:
                # If projection fails even with clean data, it points to a severe
                # numerical issue. Adding a large, but not infinite, residual can help.
                residuals_list.extend([1e6, 1e6])
        
        return np.array(residuals_list, dtype=np.float64)
    
    def create_jacobian_sparsity():
        """Create sparsity pattern for Jacobian matrix"""
        n_residuals = len(observations) * 2
        
        # Count optimizable cameras
        n_opt_cameras = len([c for c in cam_ids 
                           if not (c == first_cam_id or (anchor_second and c == second_cam_id))])
        n_params = n_opt_cameras * 6 + len(points_3d) * 3
        
        if n_params == 0:
            return None
            
        sparsity = lil_matrix((n_residuals, n_params), dtype=bool)
        
        # Track parameter offset for optimizable cameras
        cam_param_offset = {}
        offset = 0
        for cam_id in cam_ids:
            if cam_id == first_cam_id or (anchor_second and cam_id == second_cam_id):
                cam_param_offset[cam_id] = -1  # Not optimized
            else:
                cam_param_offset[cam_id] = offset
                offset += 6
        
        for obs_idx, obs in enumerate(observations):
            cam_id = obs['camera_id']
            point_idx = obs['point_idx']
            
            if point_idx >= len(points_3d):
                continue
                
            residual_start = obs_idx * 2
            
            # Camera parameters influence (only for non-anchored cameras)
            if cam_param_offset[cam_id] >= 0:
                cam_param_start = cam_param_offset[cam_id]
                sparsity[residual_start:residual_start+2, 
                        cam_param_start:cam_param_start+6] = True
            
            # Point parameters influence
            point_param_start = n_opt_cameras * 6 + point_idx * 3
            sparsity[residual_start:residual_start+2, 
                    point_param_start:point_param_start+3] = True
        
        return sparsity.tocsr()
    
    try:
        # Initial parameters
        x0 = pack_parameters()
        if len(x0) == 0:
            if verbose:
                print("[BA] All cameras anchored, no optimization needed")
            return cameras, points_3d, True
        
        jac_sparsity = create_jacobian_sparsity()
        
        result = least_squares(
            residuals, x0,
            jac_sparsity=jac_sparsity,
            loss=robust_loss,
            f_scale=loss_f_scale,
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            verbose=2 if verbose else 0
        )
        
        # Check for success or significant improvement
        if result.success or (hasattr(result, 'cost') and 
                             result.cost < result.fun.mean() * len(result.fun) * 0.95):
            # Unpack optimized parameters
            optimized_cameras, optimized_points = unpack_parameters(result.x)
            
            if verbose:
                status = "converged" if result.success else "improved significantly"
                print(f"[BA] Bundle adjustment {status}! Cost: {result.cost:.6f}")
            
            return optimized_cameras, optimized_points, True
        else:
            if verbose:
                print(f"[BA] Bundle adjustment failed: {result.message}")
            return cameras, points_3d, False
            
    except Exception as e:
        print(f"[BA] Bundle adjustment error: {e}")
        return cameras, points_3d, False


def prepare_observations_from_image_data(image_data, visited_ids, all_points_3D):
    """
    Convert image_data format to observations for BA with validation
    """
    observations = []
    
    for img_id in visited_ids:
        if img_id not in image_data:
            continue
            
        cam_data = image_data[img_id]
        if len(cam_data) < 6:
            continue
            
        R, t, K, ref_array, desc, kp = cam_data
        
        # Add observations for this camera with validation
        for kp_idx, point_idx in enumerate(ref_array):
            if (point_idx != -1 and 
                kp_idx < len(kp) and 
                isinstance(point_idx, int) and
                0 <= point_idx < len(all_points_3D)):
                
                observations.append({
                    'camera_id': img_id,
                    'point_idx': point_idx,
                    'pixel': np.array(kp[kp_idx].pt, dtype=np.float64)
                })
    
    return observations


def prepare_cameras_from_image_data(image_data, visited_ids):
    """Convert image_data format to cameras dict for BA"""
    cameras = {}
    
    for img_id in visited_ids:
        if img_id not in image_data:
            continue
            
        cam_data = image_data[img_id]
        if len(cam_data) < 6:
            continue
            
        R, t, K, ref_array, desc, kp = cam_data
        
        cameras[img_id] = {
            'R': R.astype(np.float64),
            't': t.astype(np.float64).reshape(3, 1),
            'K': K.astype(np.float64)
        }
    
    return cameras


def update_image_data_from_ba_results(image_data, optimized_cameras):
    """Update image_data with BA results"""
    for img_id, cam_opt in optimized_cameras.items():
        if img_id in image_data:
            cam_data = list(image_data[img_id])
            cam_data[0] = cam_opt['R'].astype(np.float64)
            cam_data[1] = cam_opt['t'].astype(np.float64)
            image_data[img_id] = tuple(cam_data)
    return image_data


def inter_ba(image_data, all_points_3D, visited_ids, 
                                     ba_interval=10, verbose=True):
    """Run BA every N images with gauge anchoring"""
    
    # Check if we should run BA
    if len(visited_ids) % ba_interval != 0 or len(visited_ids) < 3:
        return all_points_3D, image_data, False
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"RUNNING BUNDLE ADJUSTMENT - {len(visited_ids)} images processed")
        print(f"{'='*50}")
    
    cameras = prepare_cameras_from_image_data(image_data, visited_ids)
    observations = prepare_observations_from_image_data(image_data, visited_ids, all_points_3D)
    
    if len(cameras) < 3 or len(all_points_3D) < 10 or len(observations) < 20:
        if verbose:
            print(f"[BA] Insufficient data for BA")
        return all_points_3D, image_data, False

    optimized_cameras, optimized_points, success = global_bundle_adjustment(
        cameras, all_points_3D, observations, visited_ids,
        robust_loss='huber',
        loss_f_scale=1.5,
        max_nfev=150,
        verbose=verbose
    )
    
    if success:
        # Update image_data with optimized camera poses
        image_data = update_image_data_from_ba_results(image_data, optimized_cameras)
        
        if verbose:
            print(f"[BA] Successfully optimized {len(optimized_cameras)} cameras and {len(optimized_points)} points")
        
        return optimized_points, image_data, True
    else:
        if verbose:
            print(f"[BA] Bundle adjustment failed")
        
        return all_points_3D, image_data, False


def final_ba(image_data, all_points_3D, visited_ids, verbose=True):
    """Run final comprehensive bundle adjustment"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING FINAL BUNDLE ADJUSTMENT")
        print(f"{'='*60}")
    
    # Prepare data for BA
    cameras = prepare_cameras_from_image_data(image_data, visited_ids)
    observations = prepare_observations_from_image_data(image_data, visited_ids, all_points_3D)
    
    # Run with more iterations for final BA
    optimized_cameras, optimized_points, success = global_bundle_adjustment(
        cameras, all_points_3D, observations, visited_ids,
        robust_loss='huber',
        loss_f_scale=1.0,
        max_nfev=300,
        verbose=verbose
    )
    
    if success:
        # Update image_data with optimized camera poses
        image_data = update_image_data_from_ba_results(image_data, optimized_cameras)
        
        if verbose:
            print(f"[BA] Final BA completed successfully!")
        
        return optimized_points, image_data, True
    else:
        if verbose:
            print(f"[BA] Final BA failed")
        
        return all_points_3D, image_data, False


def compute_ba_reprojection_errors(image_data, all_points_3D, visited_ids):
    """Compute reprojection errors after BA for all cameras"""
    errors = {}
    
    for img_id in visited_ids:
        if img_id not in image_data:
            continue
            
        cam_data = image_data[img_id]
        if len(cam_data) < 6:
            continue
            
        R, t, K, ref_array, desc, kp = cam_data[:6]
        
        # Get valid 3D points for this camera
        valid_points = []
        valid_pixels = []
        
        for kp_idx, point_idx in enumerate(ref_array):
            if (point_idx != -1 and 
                kp_idx < len(kp) and
                isinstance(point_idx, int) and
                0 <= point_idx < len(all_points_3D)):
                valid_points.append(all_points_3D[point_idx][:3])
                valid_pixels.append(kp[kp_idx].pt)
        
        if len(valid_points) > 0:
            valid_points = np.array(valid_points)
            valid_pixels = np.array(valid_pixels)
            
            # Compute reprojection error
            from utils import ReprojectionError
            error = ReprojectionError(valid_pixels, R, t, K, valid_points)
            errors[img_id] = error
    
    return errors
