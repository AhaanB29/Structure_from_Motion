import cv2
import numpy as np
from itertools import combinations
from utils import resize
def select_seed_pair(img_id, path, Ks, min_inliers=100, min_baseline_ratio=0.01, top_n_pairs=10):
    """
    Robust seed pair selection for SfM (COLMAP-style).
    Uses Essential matrix + pose recovery (adapted from Reconstruction.select_baseline).
    """
    assert len(img_id) == len(Ks), "img_id and Ks length mismatch"

    sift = cv2.SIFT_create(nfeatures=5000)
    features = []

    print("[Info] Extracting SIFT features...")
    for i, p in enumerate(img_id):
        img = cv2.imread(path + p, cv2.IMREAD_GRAYSCALE)
        img,K,_ = resize(img,Ks[i])
        Ks[i] = K
        if img is None:
            print(f"Warning: Could not load {p}")
            features.append(([], None))
            continue
        kp, desc = sift.detectAndCompute(img, None)
        features.append((kp, desc if desc is not None else None))
        print(f"  {p}: {len(kp)} features")

    bf = cv2.BFMatcher()

    # Step 1: Rank candidate pairs by #matches
    print("[Info] Ranking candidate pairs...")
    pair_scores = []
    for i, j in combinations(range(len(img_id)), 2):
        kp1, desc1 = features[i]
        kp2, desc2 = features[j]
        if desc1 is None or desc2 is None: 
            continue
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) >= min_inliers:
            pair_scores.append(((i, j), len(good)))

    if not pair_scores:
        raise ValueError("No valid pairs found")

    pair_scores.sort(key=lambda x: x[1], reverse=True)
    candidate_pairs = [p for p, _ in pair_scores[:top_n_pairs]]

    # Step 2: Evaluate candidate pairs with Essential matrix + recoverPose
    best_score, best_pair = -1, None
    best_R, best_t, = None, None

    for (i, j) in candidate_pairs:
        kp1, desc1 = features[i]
        kp2, desc2 = features[j]
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        if len(pts1) < min_inliers:
            continue

        K1 = Ks[i].astype(np.float64)
        K2 = Ks[j].astype(np.float64)

        # Step 1: Estimate Fundamental Matrix in pixel coordinates
        F, mask = cv2.findFundamentalMat(pts1, pts2,
                                        method=cv2.FM_RANSAC,
                                        ransacReprojThreshold=1.0,
                                        confidence=0.999)

        if F is None or mask is None or mask.sum() < min_inliers:
            continue

        # Step 2: Compute Essential Matrix using intrinsics
        E = K2.T @ F @ K1

        # Step 3: Normalize points to camera coordinates
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None)

        # Step 4: Recover relative pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
        inliers = pose_mask.sum()
        if  inliers > best_score:
            best_pair = (img_id[i],img_id[j])
            best_score = inliers
            best_R,best_t = R,t
    print(f"[Result] Best seed pair: {best_pair}, score={best_score:.2f}")
    return best_pair, best_R, best_t
