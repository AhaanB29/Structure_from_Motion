import cv2
import numpy as np
from itertools import combinations

def select_seed_pair(img_id, path, Ks, min_inliers=100, min_baseline_ratio=0.01, top_n_pairs=10):
    """
    Robust seed pair selection for SfM (COLMAP-style).
    Args:
        img_id: list of image filenames
        path: base path to images
        Ks: list of 3x3 intrinsics for each image
        min_inliers: minimum number of inlier points required
        min_baseline_ratio: minimum baseline to depth ratio
        top_n_pairs: number of top pairs to evaluate in detail
    Returns:
        (pair, R, t, points_3d)
    """
    assert len(img_id) == len(Ks), "img_id and Ks length mismatch"

    sift = cv2.SIFT_create(nfeatures=5000)
    features = []

    print("[Info] Extracting SIFT features...")
    for i, p in enumerate(img_id):
        img = cv2.imread(path + p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {p}")
            features.append(([], None))
            continue
            
        # Resize image if too large
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            scale = 1600 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            
        kp, desc = sift.detectAndCompute(img, None)
        if desc is None:
            features.append(([], None))
            continue
            
        features.append((kp, desc))
        print(f"  {p}: {len(kp)} features")

    bf = cv2.BFMatcher()

    # Candidate pairs: rank by match count
    print("[Info] Ranking candidate pairs...")
    pair_scores = []
    
    for i, j in combinations(range(len(img_id)), 2):
        kp1, desc1 = features[i]
        kp2, desc2 = features[j]
        
        if desc1 is None or desc2 is None or len(desc1) < 50 or len(desc2) < 50:
            continue

        try:
            # Initial matching with ratio test
            matches = bf.knnMatch(desc1, desc2, k=2)
            if len(matches) < 50:
                continue
                
            # Ratio test filtering
            good = []
            for match_pair in matches:
                if len(match_pair) == 2:  # Ensure we have 2 matches
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) > 30:  # Minimum matches for consideration
                pair_scores.append(((i, j), len(good)))
                
        except Exception as e:
            print(f"Error matching {img_id[i]} and {img_id[j]}: {e}")
            continue

    if not pair_scores:
        raise ValueError("No valid pairs found with sufficient matches")

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    candidate_pairs = [p for p, _ in pair_scores[:top_n_pairs]]

    print(f"[Info] Evaluating {len(candidate_pairs)} candidate pairs...")
    best_score, best_pair = -1, None
    best_R, best_t, best_pts3d = None, None, None
    best_baseline_ratio = 0

    for (i, j) in candidate_pairs:
        try:
            kp1, desc1 = features[i]
            kp2, desc2 = features[j]
            K1, K2 = Ks[i].copy().astype(np.float64), Ks[j].copy().astype(np.float64)

            # Feature matching with ratio test
            matches = bf.knnMatch(desc1, desc2, k=2)
            good = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < min_inliers:
                continue

            # Extract matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            # Fundamental matrix estimation with RANSAC
            F, mask_F = cv2.findFundamentalMat(pts1, pts2, 
                                             method=cv2.FM_RANSAC, 
                                             ransacReprojThreshold=1.0, 
                                             confidence=0.99)
            
            if F is None or mask_F is None:
                continue

            # Filter matches using fundamental matrix
            pts1_filtered = pts1[mask_F.ravel() == 1]
            pts2_filtered = pts2[mask_F.ravel() == 1]
            
            if len(pts1_filtered) < min_inliers:
                continue

            # Convert to normalized coordinates
            pts1_norm = cv2.undistortPoints(pts1_filtered.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(pts2_filtered.reshape(-1, 1, 2), K2, None).reshape(-1, 2)

            # Essential matrix from fundamental matrix
            E = K2.T @ F @ K1
            
            # Recover pose from essential matrix
            num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, np.eye(3))
            
            if num_inliers < min_inliers:
                continue

            # Get inlier points after pose recovery
            inlier_pts1_norm = pts1_norm[mask_pose.ravel() == 1]
            inlier_pts2_norm = pts2_norm[mask_pose.ravel() == 1]
            inlier_pts1_orig = pts1_filtered[mask_pose.ravel() == 1]
            inlier_pts2_orig = pts2_filtered[mask_pose.ravel() == 1]

            if len(inlier_pts1_norm) < min_inliers:
                continue

            # Triangulation
            P1 = np.hstack((np.eye(3), np.zeros((3, 1)))).astype(np.float64)
            P2 = np.hstack((R, t)).astype(np.float64)

            # Triangulate points (use normalized coordinates)
            pts4d = cv2.triangulatePoints(P1, P2, 
                                        inlier_pts1_norm.T.astype(np.float32), 
                                        inlier_pts2_norm.T.astype(np.float32))
            
            # Convert to 3D
            pts3d = (pts4d[:3] / pts4d[3]).T

            # Filter points by cheirality (in front of both cameras)
            mask_cheirality1 = pts3d[:, 2] > 0.1  # Points in front of camera 1
            pts_cam2 = (R @ pts3d.T + t).T
            mask_cheirality2 = pts_cam2[:, 2] > 0.1  # Points in front of camera 2
            
            mask_cheirality = mask_cheirality1 & mask_cheirality2
            
            if np.sum(mask_cheirality) < min_inliers:
                continue

            pts3d_good = pts3d[mask_cheirality]
            pts1_good = inlier_pts1_orig[mask_cheirality]
            pts2_good = inlier_pts2_orig[mask_cheirality]

            # Reprojection error check
            # Project 3D points back to both images
            pts_cam1 = pts3d_good  # Already in camera 1 coordinates
            pts_cam2 = (R @ pts3d_good.T + t).T

            # Project to image coordinates
            pts_img1_proj = (K1 @ pts_cam1.T).T
            pts_img1_proj = pts_img1_proj[:, :2] / pts_img1_proj[:, 2:3]

            pts_img2_proj = (K2 @ pts_cam2.T).T  
            pts_img2_proj = pts_img2_proj[:, :2] / pts_img2_proj[:, 2:3]

            # Calculate reprojection errors
            err1 = np.linalg.norm(pts1_good - pts_img1_proj, axis=1)
            err2 = np.linalg.norm(pts2_good - pts_img2_proj, axis=1)
            
            # Filter by reprojection error
            reproj_threshold = 2.0
            mask_reproj = (err1 < reproj_threshold) & (err2 < reproj_threshold)
            
            if np.sum(mask_reproj) < min_inliers:
                continue

            pts3d_final = pts3d_good[mask_reproj]

            # Calculate baseline ratio (baseline / mean_depth)
            baseline = np.linalg.norm(t)
            mean_depth = np.mean(pts3d_final[:, 2])
            baseline_ratio = baseline / mean_depth if mean_depth > 0 else 0

            # Reject pairs with poor geometry
            if baseline_ratio < min_baseline_ratio:
                continue

            # Calculate triangulation angles for better geometry assessment
            C1 = np.zeros(3)  # Camera 1 center
            C2 = -R.T @ t.flatten()  # Camera 2 center
            
            rays1 = pts3d_final - C1
            rays2 = pts3d_final - C2.reshape(1, -1)
            
            # Normalize rays
            rays1_norm = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
            rays2_norm = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)
            
            # Calculate parallax angles
            cos_angles = np.sum(rays1_norm * rays2_norm, axis=1)
            cos_angles = np.clip(cos_angles, -1, 1)
            angles = np.degrees(np.arccos(np.abs(cos_angles)))
            
            # Filter points with good parallax (> 2 degrees)
            good_parallax = angles > 2.0
            if np.sum(good_parallax) < min_inliers:
                continue

            pts3d_final = pts3d_final[good_parallax]
            mean_angle = np.mean(angles[good_parallax])

            # Score combines number of points, baseline ratio, and mean parallax angle
            score = len(pts3d_final) * baseline_ratio * (mean_angle / 30.0)  # Normalize angle
            
            print(f"  Pair {img_id[i]}-{img_id[j]}: {len(pts3d_final)} points, "
                  f"baseline_ratio={baseline_ratio:.4f}, mean_angle={mean_angle:.1f}°, score={score:.2f}")

            if score > best_score:
                best_score = score
                best_pair = (img_id[i], img_id[j])
                best_R, best_t, best_pts3d = R.copy(), t.copy(), pts3d_final.copy()
                best_baseline_ratio = baseline_ratio

        except Exception as e:
            print(f"Error processing pair {img_id[i]}-{img_id[j]}: {e}")
            continue

    if best_pair is None:
        raise ValueError("No suitable seed pair found. Try reducing min_inliers or min_baseline_ratio.")

    print(f"[Result] Best seed pair: {best_pair}")
    print(f"  Points: {len(best_pts3d)}")
    print(f"  Baseline ratio: {best_baseline_ratio:.4f}")
    print(f"  Score: {best_score:.2f}")
    print(f"  Translation magnitude: {np.linalg.norm(best_t):.3f}")
    
    return best_pair, best_R, best_t, best_pts3d


def visualize_seed_pair_matches(img_id, path, pair, Ks, max_matches=100):
    """
    Visualize matches for the selected seed pair (optional utility function)
    """
    try:
        import matplotlib.pyplot as plt
        
        idx1 = img_id.index(pair[0])
        idx2 = img_id.index(pair[1])
        
        img1 = cv2.imread(path + pair[0])
        img2 = cv2.imread(path + pair[1])
        
        if img1 is None or img2 is None:
            print("Could not load images for visualization")
            return
            
        # Convert to RGB for matplotlib
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        for img in [img1, img2]:
            h, w = img.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
        
        # Extract features and matches
        sift = cv2.SIFT_create(nfeatures=5000)
        kp1, desc1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
        kp2, desc2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        good = good[:max_matches]  # Limit for visualization
        
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(img_matches)
        plt.title(f'Seed Pair Matches: {pair[0]} - {pair[1]} ({len(good)} matches)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization error: {e}")