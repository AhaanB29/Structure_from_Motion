import cv2
import numpy as np
from itertools import combinations

def select_seed_pair(img_id, path, Ks, min_inliers=100, min_baseline_ratio=0.01, top_n_pairs=5):
    """
    Robust seed pair selection for SfM (COLMAP-style).
    Args:
        img_id: list of image filenames
        Ks: list of 3x3 intrinsics for each image
        path: base path to images
    Returns:
        (pair, R, t, points_3d)
    """
    assert len(img_id) == len(Ks), "img_id and Ks length mismatch"

    sift = cv2.SIFT_create(nfeatures=5000)
    features = []

    print("[Info] Extracting SIFT features...")
    for p in img_id:
        img = cv2.imread(path + p, cv2.IMREAD_GRAYSCALE)
        kp, desc = sift.detectAndCompute(img, None)
        features.append((kp, desc))

    bf = cv2.BFMatcher()

    # Candidate pairs: rank by match count
    print("[Info] Ranking candidate pairs...")
    pair_scores = []
    for i, j in combinations(range(len(img_id)), 2):
        desc1, desc2 = features[i][1], features[j][1]
        if desc1 is None or desc2 is None:
            continue

        # Ratio test
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Mutual check (cross check like COLMAP)
        matches_rev = bf.knnMatch(desc2, desc1, k=2)
        good_rev = [m for m, n in matches_rev if m.distance < 0.75 * n.distance]
        good_idx1 = set([m.queryIdx for m in good])
        good_idx2 = set([m.trainIdx for m in good_rev])
        mutual = [m for m in good if (m.trainIdx in good_idx2)]

        if len(mutual) > 15:
            pair_scores.append(((i, j), len(mutual)))

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    candidate_pairs = [p for p, _ in pair_scores[:top_n_pairs]]

    print(f"[Info] Evaluating {len(candidate_pairs)} candidate pairs...")
    best_score, best_pair = -1, None
    best_R, best_t, best_pts3d = None, None, None

    for (i, j) in candidate_pairs:
        kp1, desc1 = features[i]
        kp2, desc2 = features[j]

        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < min_inliers:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        K1, K2 = Ks[i], Ks[j]

        # Normalize with intrinsics
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, np.eye(3),
                                       method=cv2.RANSAC, prob=0.999, threshold=1e-3)
        if E is None:
            continue

        inlier_pts1 = pts1_norm[mask.ravel() == 1]
        inlier_pts2 = pts2_norm[mask.ravel() == 1]
        if len(inlier_pts1) < min_inliers:
            continue

        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, inlier_pts1, inlier_pts2)
        inlier_pts1, inlier_pts2 = inlier_pts1[mask_pose.ravel() == 1], inlier_pts2[mask_pose.ravel() == 1]

        # Triangulation
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # Cheirality: keep points in front of both cameras
        mask_ch = (pts3d[:, 2] > 0) & ((R @ pts3d.T + t).T[:, 2] > 0)
        pts3d = pts3d[mask_ch]

        if len(pts3d) < min_inliers:
            continue

        # Reprojection error check
        proj1 = pts3d[:, :2] / pts3d[:, 2:3]
        proj2 = (R @ pts3d.T + t).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        err1 = np.linalg.norm(inlier_pts1[:len(proj1)] - proj1, axis=1)
        err2 = np.linalg.norm(inlier_pts2[:len(proj2)] - proj2, axis=1)
        good_mask = (err1 < 2.0) & (err2 < 2.0)
        pts3d = pts3d[good_mask]

        if len(pts3d) < min_inliers:
            continue

        # Baseline ratio
        baseline = np.linalg.norm(t)
        depth_mean = np.mean(pts3d[:, 2])
        baseline_ratio = baseline / depth_mean

        score = len(pts3d) * baseline_ratio
        if baseline_ratio > min_baseline_ratio and score > best_score:
            best_score = score
            best_pair = (img_id[i], img_id[j])
            best_R, best_t, best_pts3d = R, t, pts3d

    if best_pair is None:
        raise ValueError("No suitable seed pair found.")

    print(f"[Result] Seed pair: {best_pair}, #Points: {len(best_pts3d)}, "
          f"Baseline ratio: {baseline_ratio:.4f}, Score: {best_score:.2f}")
    return best_pair, best_R, best_t, best_pts3d
