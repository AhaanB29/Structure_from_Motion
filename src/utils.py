
import cv2 
import numpy as np
import pandas as pd

def get_colors(img_nm, pts, path):
    img1 = cv2.imread(path + img_nm, cv2.IMREAD_COLOR)
    h, w, _ = img1.shape
    colors = []
    
    for x, y in pts.reshape((-1, 2)):
        x = int(round(x))
        y = int(round(y))
        
        if 0 <= x < w and 0 <= y < h:
            clr = img1[y, x]  
            colors.append(clr / 255.0)  
        else:
            colors.append((0.0, 0.0, 0.0))  
    
    return np.array(colors, dtype=np.float32)


def Camera_Params(file_path, img_id):
    data = pd.read_csv(file_path)
    # Extract corresponding matrices for each image ID
    all_R, all_t, all_k = [], [], []
    for i in img_id:
        row = data[data['image_name'] == i]
        if row.empty:
            print(f"Warning: {i} not found in CSV! Skipping.")
            continue
        R =np.array((data[data['image_name'] == i]['rotation_matrix']).values[0].split(';'),dtype=np.float32).reshape(3,3)
        t = np.array((data[data['image_name']== i]['translation_vector']).values[0].split(';'),dtype=np.float32).reshape(1,3)
        K = np.array((data[data['image_name']== i]['calibration_matrix']).values[0].split(';'),dtype=np.float32).reshape(3,3)

        all_R.append(R)
        all_t.append(t)
        all_k.append(K)
    return np.stack(all_R), np.stack(all_t), np.stack(all_k)



def NormalizePts(x):
    mus = x[:,:2].mean(axis=0)
    sigma = x[:,:2].std()
    scale = np.sqrt(2.) / sigma

    transMat = np.array([[1,0,mus[0]],[0,1,mus[1]],[0,0,1]])
    scaleMat = np.array([[scale,0,0],[0,scale,0],[0,0,1]])

    T = scaleMat.dot(transMat)

    xNorm = T.dot(x.T).T

    return xNorm, T
def ExtractCameraPoses(E): 
    u,d,v = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    Rs, Cs = np.zeros((4,3,3)), np.zeros((4,3))
    
    t = u[:,-1]
    R1 = u.dot(W.dot(v))
    R2 = u.dot(W.T.dot(v))

    if np.linalg.det(R1) < 0: 
        R1 = R1 * -1

    if np.linalg.det(R2) < 0: 
        R2 = R2 * -1
    
    return R1,R2,t

def GetImageMatches(img1,img2):
    sift=cv2.SIFT_create(nfeatures=5000)
    kp1, desc1 = sift.detectAndCompute(img1,None)
    kp2, desc2 = sift.detectAndCompute(img2,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)
    return kp1,desc1,kp2,desc2,matches


def BaseTriangulation(kp1, kp2, mask, K1, K2, R1, t1, R2, t2, matches, all_points_3D,
                      parallax_deg_thresh=0.3, reproj_thresh=8.0, max_depth=50.0):
    """
    Triangulate matches between two images and return new 3D points and reference arrays.

    Inputs:
      kp1, kp2      : lists of cv2.KeyPoint for image1 and image2
      mask          : boolean mask (len == len(matches)) from geometric verification (e.g. essential/fundamental)
      K1, K2        : intrinsics (3x3)
      R1, t1        : rotation (3x3) and translation (3x1) of image1 (camera1 in world)
      R2, t2        : rotation (3x3) and translation (3x1) of image2
      matches       : list of cv2.DMatch between img1 (queryIdx) and img2 (trainIdx)
      all_points_3D : list of existing 3D points (each element [x,y,z,r,g,b] or similar)
      parallax_deg_thresh : minimum parallax angle in degrees (default 1.5)
      reproj_thresh : max allowed reprojection error in pixels (default 4.0)
      max_depth     : absolute depth cutoff in camera1 frame (units of your reconstruction)

    Returns:
      pts3d_good    : (M,3) ndarray of accepted 3D points
      ref1          : list of length len(kp1) with 3D indices or -1
      ref2          : list of length len(kp2) with 3D indices or -1
      all_points_3D : updated list with appended points (same format as input)
    """

    # Ensure t shapes
    t1 = np.asarray(t1).reshape(3,1)
    t2 = np.asarray(t2).reshape(3,1)

    # Build image points arrays from matches and mask
    if mask is None:
        mask = np.ones(len(matches), dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)

    # Filter matches by mask
    valid_matches = [m for m, msk in zip(matches, mask) if bool(msk)]
    if len(valid_matches) < 2:
        # not enough matches to triangulate
        ref1 = [-1] * len(kp1)
        ref2 = [-1] * len(kp2)
        return np.zeros((0,3)), ref1, ref2, all_points_3D

    img1_pts = np.array([kp1[m.queryIdx].pt for m in valid_matches], dtype=np.float64)  # (N,2)
    img2_pts = np.array([kp2[m.trainIdx].pt for m in valid_matches], dtype=np.float64)  # (N,2)

    # Build projection matrices in pixel coords
    P1 = K1 @ np.hstack((R1, t1))   # (3x4)
    P2 = K2 @ np.hstack((R2, t2))

    # Triangulate: cv2.triangulatePoints expects 2xN arrays
    pts4d = cv2.triangulatePoints(P1, P2, img1_pts.T.astype(np.float64), img2_pts.T.astype(np.float64))  # (4,N)
    pts3d = (pts4d[:3, :] / pts4d[3, :]).T   # (N,3)

    # Masks: finite, cheirality, reprojection, parallax, depth
    mask_finite = np.isfinite(pts3d).all(axis=1)

    # Cheirality: point in front of both cameras
    pts_cam1 = (R1 @ pts3d.T + t1).T   # shape (N,3)
    pts_cam2 = (R2 @ pts3d.T + t2).T
    mask_front = (pts_cam1[:,2] > 1e-6) & (pts_cam2[:,2] > 1e-6)

    # Reprojection error: project back using P1,P2 and compute Euclidean pixel error
    homog = np.hstack((pts3d, np.ones((pts3d.shape[0],1))))
    proj1 = (P1 @ homog.T).T  # (N,3)
    proj2 = (P2 @ homog.T).T
    proj1_xy = proj1[:, :2] / proj1[:, 2:3]
    proj2_xy = proj2[:, :2] / proj2[:, 2:3]
    err1 = np.linalg.norm(proj1_xy - img1_pts, axis=1)
    err2 = np.linalg.norm(proj2_xy - img2_pts, axis=1)
    mask_reproj = (err1 < reproj_thresh) & (err2 < reproj_thresh)

    # Parallax angle check (angle between two viewing rays)
    # camera centers:
    C1 = -R1.T @ t1   # (3,1)
    C2 = -R2.T @ t2
    rays1 = pts3d - C1.ravel()
    rays2 = pts3d - C2.ravel()
    # normalize
    nr1 = np.linalg.norm(rays1, axis=1)
    nr2 = np.linalg.norm(rays2, axis=1)
    # avoid division by zero
    valid_len = (nr1 > 1e-8) & (nr2 > 1e-8)
    cosang = np.zeros(pts3d.shape[0], dtype=np.float64)
    cosang[valid_len] = np.sum(rays1[valid_len] * rays2[valid_len], axis=1) / (nr1[valid_len] * nr2[valid_len])
    cosang = np.clip(cosang, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosang))
    mask_angle = angles_deg > parallax_deg_thresh

    # Depth bound (in camera1 frame)
    mask_depth = pts_cam1[:,2] < max_depth

    mask_all = mask_finite & mask_front & mask_reproj & mask_angle & mask_depth

    # Apply mask
    pts3d_good = pts3d[mask_all]
    img1_good = img1_pts[mask_all]
    img2_good = img2_pts[mask_all]
    kept_matches = [m for m, keep in zip(valid_matches, mask_all) if keep]

    # Prepare ref arrays (map keypoint index -> 3D point index)
    ref1 = [-1] * len(kp1)
    ref2 = [-1] * len(kp2)

    # Append each accepted 3D point to global list and set refs
    start_idx = len(all_points_3D)
    for i, m in enumerate(kept_matches):
        X = pts3d_good[i]
        # color placeholder white (you can change to actual sampled color)
        all_points_3D.append([float(X[0]), float(X[1]), float(X[2]), 255, 255, 255])
        global_idx = start_idx + i
        ref1[m.queryIdx] = global_idx
        ref2[m.trainIdx] = global_idx

    # If nothing kept, return empty arrays but keep ref arrays
    if pts3d_good.shape[0] == 0:
        return np.zeros((0,3)), ref1, ref2, all_points_3D

    return pts3d_good, ref1, ref2, all_points_3D

def TransformCoordPts(X,R,t): 
    ''' X : 3D points
        R : Rotation matrix
        t : translation Vector
    '''
    return (R.dot(X.T)+t).T

def CountFrontOfBothCameras(X, R, t): 
    isfrontcam1 = X[:,-1] > 0
    isfrontcam2 = TransformCoordPts(X,R,t)[:,-1] > 0
    
    return np.sum(isfrontcam1 & isfrontcam2)

def DisambiguateCameraPose(configSet):  
    maxfrontpts = -1 
    
    for R,t,pts3d in configSet: 
        count = CountFrontOfBothCameras(pts3d,R,t[:,np.newaxis])
        
        if count > maxfrontpts: 
            maxfrontpts = count
            bestR,bestt = R,t
    
    return bestR,bestt,maxfrontpts

def SeedPair_PoseEstimation(kp1,desc1,kp2,desc2,K1,K2,R_0,t_0,matches):
    
    pts1 = [kp1[m.queryIdx].pt for m in matches]
    pts2 = [kp2[m.trainIdx].pt for m in matches]
    pts1,pts2 = np.array(pts1),np.array(pts2)
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,1.0,0.99)
    E = K2.T.dot(F.dot(K1))
    R1,R2,t = ExtractCameraPoses(E)
    configSet = [None,None,None,None]
    
    configSet[0] = (R1,t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R1,t[:,np.newaxis],K2))
    configSet[1] = (R1,-t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R1,-t[:,np.newaxis],K2))
    configSet[2] = (R2,t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R2,t[:,np.newaxis],K2))
    configSet[3] = (R2,-t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R2,-t[:,np.newaxis],K2))

    R,t,count  = DisambiguateCameraPose(configSet)

    return R,t,mask

def ReprojectionError(img1pts, R, t, K, pts3d):
    # pts3d: (N, 3) in world coords
    pts_cam = (R @ pts3d.T) + t.reshape(3, 1)       # (3, N)
    pts_img_h = K @ pts_cam                         # (3, N)
    pts_img = (pts_img_h[:2] / pts_img_h[2]).T      # (N, 2)
    
    # Euclidean distance between measured 2D points and projected points
    return np.mean(np.linalg.norm(img1pts - pts_img, axis=1))

def Correspondence2D_3D(img_data,img,all_points3D):
    sift=cv2.SIFT_create(nfeatures=8000)
    kp_new, desc_new = sift.detectAndCompute(img,None)
    matcher = cv2.BFMatcher(crossCheck=True)
    updated_3D_pts = []
    filtered_kp = []
    for id in img_data.keys():
        _,_,_,ref_arr,desc_org,_ =img_data[id]
        matches = matcher.match(desc_new, desc_org)
        matches = sorted(matches, key = lambda x:x.distance)
        img_new_idx = [m.queryIdx for m in matches]
        img_new_pts = [kp_new[m.queryIdx].pt for m in matches]
        for m in matches:
            idx_3d = ref_arr[m.trainIdx]
            if idx_3d != -1 :
                updated_3D_pts.append(all_points3D[idx_3d][:3])
                filtered_kp.append(kp_new[m.queryIdx].pt) 
    return np.array(filtered_kp), np.array(updated_3D_pts),desc_new,kp_new

def Triangulate2Views(img1pts,img2pts,R1,t1,K1,R2,t2,K2):
    if R1 is None: 
        R1 = np.eye(3) 
    if t1 is None: 
        t1 = np.zeros((3,1))

    
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K1).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K2).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]
    
    print (R1.shape, t1.shape, R2.shape, t2.shape)
    pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
    
    return pts3d

def Toply(pts,filename='out.ply'): 
    f = open(filename,'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(pts.shape[0]))
    
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    
    f.write('end_header\n')
    
    for pt in pts: 
        f.write('{} {} {} 255 255 255\n'.format(pt[0],pt[1],pt[2]))
    f.close()

def select_next_image(image_data, all_points_3D, unregistered_ids, path, min_corr=30):
    """
    Selects the next best image to register into the model.
    Based on number of valid 2D–3D correspondences.
    """
    best_score = -1
    best_img_id = None
    best_data = None

    for img_id in unregistered_ids:
        img = cv2.imread(path + img_id)
        matched_2D_pts, matched_3D_pts, desc_new, kp_new = Correspondence2D_3D(image_data, img, all_points_3D)

        score = matched_3D_pts.shape[0]  # number of correspondences
        if score > best_score and score >= min_corr:
            best_score = score
            best_img_id = img_id
            best_data = (matched_2D_pts, matched_3D_pts, desc_new, kp_new)

    return best_img_id, best_data[0],best_data[1],best_data[2],best_data[3]