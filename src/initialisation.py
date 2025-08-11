import cv2
import numpy as np
from itertools import combinations

def select_seed_pair(path,img_id, Ks, min_inliers=50, min_baseline_ratio=0.01, top_n_pairs=50):
    """
    Selects a good seed pair for SfM using per-image intrinsics.

    Args:
        img_id: list of image file paths.
        Ks: list of 3x3 intrinsic matrices, same order as img_id.
        min_inliers: minimum number of inlier matches.
        min_baseline_ratio: baseline/depth ratio threshold.
        top_n_pairs: limit number of candidate pairs to evaluate.
    Returns:
        (idx1, idx2, R, t, points_3d)
    """
    assert len(img_id) == len(Ks), "img_id and Ks must have same length"
    
    sift = cv2.SIFT_create(nfeatures=4000)
    features = []

    print("[Info] Extracting features...")
    for p in img_id:
        img = cv2.imread(path + p, cv2.IMREAD_GRAYSCALE)
        kp, desc = sift.detectAndCompute(img, None)
        features.append((kp, desc))

    print("[Info] Ranking pairs by raw match count...")
    matcher = cv2.BFMatcher()
    pair_scores = []
    for i, j in combinations(range(len(img_id)), 2):
        desc1 = features[i][1]
        desc2 = features[j][1]
        if desc1 is None or desc2 is None:
            continue
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        pair_scores.append(((i, j), len(good)))

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    candidate_pairs = [p for p, _ in pair_scores[:top_n_pairs]]

    print(f"[Info] Evaluating {len(candidate_pairs)} candidate pairs...")
    best_score = -1
    best_pair = None
    best_R, best_t, best_pts3d = None, None, None

    for (i, j) in candidate_pairs:
        kp1, desc1 = features[i]
        kp2, desc2 = features[j]

        matches = matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < min_inliers:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        K1 = Ks[i]
        K2 = Ks[j]

        E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue

        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        if len(inlier_pts1) < min_inliers:
            continue

        _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K1)

        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K2 @ np.hstack((R, t))
        pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # Cheirality check
        front1 = pts3d[:, 2] > 0
        front2 = (R @ pts3d.T + t).T[:, 2] > 0
        mask_ch = front1 & front2
        pts3d = pts3d[mask_ch]

        if len(pts3d) < min_inliers:
            continue

        baseline = np.linalg.norm(t)
        depth_mean = np.mean(pts3d[:, 2])
        baseline_ratio = baseline / depth_mean

        score = len(pts3d) * baseline_ratio
        if baseline_ratio > min_baseline_ratio and score > best_score:
            best_score = score
            best_pair = (i, j)
            best_R, best_t = R, t
            best_pts3d = pts3d

    if best_pair is None:
        raise ValueError("No suitable seed pair found.")

    print(f"[Result] Seed pair: {best_pair}, #Points: {len(best_pts3d)}, Baseline ratio: {baseline_ratio:.4f}")
    return best_pair, best_R, best_t, best_pts3d
