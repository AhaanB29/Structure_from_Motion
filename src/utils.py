import cv2 
import numpy as np
import pandas as pd
import open3d as o3d
sift=cv2.SIFT_create(nfeatures=5000)

def resize(img, K, max_dim=1600):
    """
    Resize image and update camera intrinsics accordingly
    """
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Scale the intrinsic matrix
        K_new = K.copy().astype(np.float64)
        K_new[0, 0] *= scale  # fx
        K_new[1, 1] *= scale  # fy  
        K_new[0, 2] *= scale  # cx
        K_new[1, 2] *= scale  # cy
        
        return img_resized, K_new, scale
    else:
        return img, K.copy(), 1.0

def get_colors(img_nm, pts, path,K):
    img1 = cv2.imread(path + img_nm, cv2.IMREAD_COLOR)
    img1,_,_ = resize(img1,K)
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
        #i = i[:-4]
        row = data[data['image_id'] == i]
        if row.empty:
            print(f"Warning: {i} not found in CSV! Skipping.")
            continue
        R =np.array((data[data['image_id'] == i]['rotation_matrix']).values[0].split(' '),dtype=np.float32).reshape(3,3)
        t = np.array((data[data['image_id']== i]['translation_vector']).values[0].split(' '),dtype=np.float32).reshape(1,3)
        K = np.array((data[data['image_id']== i]['camera_intrinsics']).values[0].split(' '),dtype=np.float32).reshape(3,3)

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
    
    kp1, desc1 = sift.detectAndCompute(img1,None)
    kp2, desc2 = sift.detectAndCompute(img2,None)

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key = lambda x:x.distance)
    return kp1,desc1,kp2,desc2,matches


def BaseTriangulation(kp1, kp2, mask, K1, K2, R1, t1, R2, t2, matches,
                      parallax_deg_thresh=3.0, reproj_thresh=4.0, max_depth=10.0):
    """
    Triangulate matches between two images and return new 3D points and reference arrays.
    
    Fixed version to avoid diverging effects.
    """
    # Ensure t shapes - more robust
    t1 = np.asarray(t1).reshape(-1, 1)
    t2 = np.asarray(t2).reshape(-1, 1)
    
    # Ensure R shapes
    R1 = np.asarray(R1).reshape(3, 3)
    R2 = np.asarray(R2).reshape(3, 3)

    # Build image points arrays from matches and mask
    if mask is None:
        mask = np.ones(len(matches), dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)

    # Filter matches by mask
    valid_matches = [m for m, msk in zip(matches, mask) if bool(msk)]
    
    # Early exit if insufficient matches
    if len(valid_matches) < 8:  # Need minimum points for robust triangulation
        print("Not enough matches to triangulate")
        return np.zeros((0,3)), [], [], []

    img1_pts = np.array([kp1[m.queryIdx].pt for m in valid_matches], dtype=np.float64)
    img2_pts = np.array([kp2[m.trainIdx].pt for m in valid_matches], dtype=np.float64)

    # Build projection matrices
    P1 = K1 @ np.hstack((R1, t1))
    P2 = K2 @ np.hstack((R2, t2))

    # Triangulate
    pts4d = cv2.triangulatePoints(P1, P2, img1_pts.T, img2_pts.T)
    
     
    # Convert to 3D
    pts3d = (pts4d[:3, :] / pts4d[3, :]).T

    print("Avg Z values:", np.mean(np.array(pts3d[2]).reshape(1,3)))
    # 1. Finite check
    mask_finite = np.isfinite(pts3d).all(axis=1)

    # 2. Cheirality check - points in front of both cameras
    pts_cam1 = (R1 @ pts3d.T + t1).T
    pts_cam2 = (R2 @ pts3d.T + t2).T
    mask_front =  (pts_cam2[:,2] > -1e4)  # More conservative

    # 3. Depth check
    mask_depth = (pts_cam1[:,2] < max_depth) & (pts_cam2[:,2] < max_depth)
    
    # 4. Reprojection error check
    homog = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    proj1 = (P1 @ homog.T).T
    proj2 = (P2 @ homog.T).T
    
    # Check for valid projections
    mask_proj_valid = (np.abs(proj1[:, 2]) > 1e-8) & (np.abs(proj2[:, 2]) > 1e-8)
    
    proj1_xy = np.zeros((pts3d.shape[0], 2))
    proj2_xy = np.zeros((pts3d.shape[0], 2))
    proj1_xy[mask_proj_valid] = proj1[mask_proj_valid, :2] / proj1[mask_proj_valid, 2:3]
    proj2_xy[mask_proj_valid] = proj2[mask_proj_valid, :2] / proj2[mask_proj_valid, 2:3]
    
    err1 = np.linalg.norm(proj1_xy - img1_pts, axis=1)
    err2 = np.linalg.norm(proj2_xy - img2_pts, axis=1)
    mask_reproj = (err1 < reproj_thresh) & (err2 < reproj_thresh) 

    # 5. Parallax angle check - CRITICAL for stability
    C1 = -R1.T @ t1  # Camera centers
    C2 = -R2.T @ t2
    
    rays1 = pts3d - C1.ravel()
    rays2 = pts3d - C2.ravel()
    
    # Normalize rays
    nr1 = np.linalg.norm(rays1, axis=1)
    nr2 = np.linalg.norm(rays2, axis=1)
    
    valid_len = (nr1 > 1e-8) & (nr2 > 1e-8)
    cosang = np.full(pts3d.shape[0], -2.0, dtype=np.float64)  # Invalid by default
    
    if np.any(valid_len):
        rays1_norm = rays1[valid_len] / nr1[valid_len, np.newaxis]
        rays2_norm = rays2[valid_len] / nr2[valid_len, np.newaxis]
        cosang[valid_len] = np.sum(rays1_norm * rays2_norm, axis=1)
    
    cosang = np.clip(cosang, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(np.abs(cosang)))  # Use abs for angle
    mask_angle = (angles_deg > parallax_deg_thresh) & valid_len

    # 6. Distance sanity check
    dist_from_cam1 = np.linalg.norm(pts3d - C1.ravel(), axis=1)
    dist_from_cam2 = np.linalg.norm(pts3d - C2.ravel(), axis=1)
    baseline = np.linalg.norm(C1 - C2)
    mask_distance = (dist_from_cam1 > 0.1 * baseline) & (dist_from_cam2 > 0.1 * baseline)

    # Combine all masks
    mask_all = mask_finite& mask_reproj & mask_depth & mask_front#& mask_distance #& mask_angle 

    # Apply final mask
    pts3d_good = pts3d[mask_all]
    img1_good = img1_pts[mask_all]
    img2_good = img2_pts[mask_all]
    valid_matches_good = [valid_matches[i] for i in range(len(valid_matches)) if mask_all[i]]

    if pts3d_good.shape[0] == 0:
        print("No good 3D points after filtering")
        return np.zeros((0,3)), [], [], []
    
    return pts3d_good, img1_good, img2_good, valid_matches_good

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
    E = K2.T @ F @ K1
    configSet = [None,None,None,None]
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K1)
    # R1,R2,t = ExtractCameraPoses(E)
    # configSet[0] = (R1,t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R1,t[:,np.newaxis],K2))
    # configSet[1] = (R1,-t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R1,-t[:,np.newaxis],K2))
    # configSet[2] = (R2,t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R2,t[:,np.newaxis],K2))
    # configSet[3] = (R2,-t,Triangulate2Views(pts1[mask],pts2[mask],R_0,t_0,K1,R2,-t[:,np.newaxis],K2))

    # R,t,count  = DisambiguateCameraPose(configSet)
    return R,t,mask

def ReprojectionError(img1pts, R, t, K, pts3d):
    # pts3d: (N, 3) in world coords
    pts_cam = (R @ pts3d.T) + t.reshape(3, 1)     # (3, N)
    pts_h = K @ pts_cam
    pts_reproj = (pts_h[:2] / pts_h[2]).T     # (N, 2) 
    return np.mean(np.linalg.norm(img1pts - pts_reproj, axis=1))

def Correspondence2D_3D(image_data, img, all_points3D):

    kp_new, desc_new = sift.detectAndCompute(img, None)

    # Build the 'search database' of descriptors that already have 3D points
    desc_db = []
    pts3d_db = []

    for img_id, (R, t, K, ref_arr, desc, kp) in image_data.items():
        ref_arr = np.array(ref_arr)
        valid_idxs = np.where(ref_arr != -1)[0]  # features with associated 3D points

        if len(valid_idxs) == 0:
            continue

        desc_db.append(desc[valid_idxs])
        pts3d_db.append(np.array([all_points3D[i][:3] for i in ref_arr[valid_idxs]]))

    if len(desc_db) == 0:
        return np.array([]), np.array([]), desc_new, kp_new

    # Concatenate all descriptors and corresponding 3D points
    desc_db = np.vstack(desc_db)
    pts3d_db = np.vstack(pts3d_db)

    # Match new image descriptors to the 3D-point descriptors
    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(desc_new, desc_db)
    matches = sorted(matches, key=lambda x: x.distance)

    # Build matched 2D and 3D point arrays
    matched_2D = np.array([kp_new[m.queryIdx].pt for m in matches], dtype=np.float32)
    matched_3D = np.array([pts3d_db[m.trainIdx] for m in matches], dtype=np.float32)

    return matched_2D, matched_3D, desc_new, kp_new

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
    pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
    
    return pts3d

def Toply(pts,clrs, filename='out.ply'):
    clrs = np.clip(clrs * 255.0, 0, 255).astype(np.uint8) 
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
    
    for pt,clr in zip(pts,clrs): 
        f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],clr[0],clr[1],clr[2]))
    f.close()

def select_next_image(image_data, all_points_3D, unregistered_ids, path,Ks,img_ids,min_corr=30):
    """
    Selects the next best image to register into the model.
    Based on number of valid 2D–3D correspondences.
    """
    best_score = -1
    best_img_id = None
    best_data = None
    best_img = None
    K_best=None

    for img_id in unregistered_ids:
        img = cv2.imread(path + img_id)
        K = Ks[img_ids.index(img_id)]
        img,K,_ = resize(img,K) 
        matched_2D_pts, matched_3D_pts, desc_new, kp_new = Correspondence2D_3D(image_data, img, all_points_3D)

        score = matched_3D_pts.shape[0]  # number of correspondences
        if score > best_score and score >= min_corr:
            best_score = score
            best_img_id = img_id
            best_data = (matched_2D_pts, matched_3D_pts, desc_new, kp_new)
            best_img = img.copy()
            K_best = K
            #break
    if best_img is None:
        return None,None,None,None,None,None
    
    
    # img_vis = best_img.copy()
    # for (x, y) in best_data[0]:  # matched_2D_pts
    #     img_vis = cv2.circle(img_vis, (int(x), int(y)), 8, (0, 255, 0), -1)

    # plt.figure(figsize=(10, 6))
    # plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    # plt.title(f"2D–3D Matches on Image {best_img_id} ({best_score} correspondences)")
    # plt.axis('off')
    # plt.show()
    # plt.close('all')
    return best_img_id, best_data[0],best_data[1],best_data[2],best_data[3],K_best

def camera_pose_vis(K, width, height, scale=2, R=np.eye(3), t=np.zeros((3, 1))):
    """
    Create a square-based camera pyramid (like COLMAP) as a LineSet in Open3D,
    along with camera coordinate axes.

    Parameters:
        K      : (3x3) intrinsic matrix
        width  : image width
        height : image height
        scale  : scale factor for pyramid size
        R, t   : SfM extrinsics (world -> camera)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # corners of the image plane in camera coords (normalized & scaled)
    corners = np.array([
        [(0 - cx) / fx, (0 - cy) / fy, 1.0],          # top-left
        [(width - cx) / fx, (0 - cy) / fy, 1.0],      # top-right
        [(width - cx) / fx, (height - cy) / fy, 1.0], # bottom-right
        [(0 - cx) / fx, (height - cy) / fy, 1.0],     # bottom-left
    ])
    corners *= scale

    # Camera center (origin in cam coords)
    cam_center = np.zeros((1, 3))

    # Pyramid vertices in camera coordinates
    points_cam = np.vstack((cam_center, corners))  # shape (5,3) → [0]=center, [1..4]=corners

    # --- Convert SfM extrinsics to world pose ---
    # SfM: X_c = R * X_w + t
    # Camera center in world: C = -R.T @ t
    R_w = R.T
    t_w = -R.T @ t.reshape(3, 1)

    # Build homogeneous transform (camera->world)
    Rt = np.hstack((R_w, t_w))
    Rt = np.vstack((Rt, [0, 0, 0, 1]))

    # Transform pyramid points into world coordinates
    points_world = (Rt @ np.hstack((points_cam, np.ones((5, 1)))).T).T[:, :3]

    # --- Edges of pyramid (square base) ---
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],   # tip to base corners
        [1, 2], [2, 3], [3, 4], [4, 1]    # base square
    ]
    colors = [[0, 0, 1] for _ in lines]  # blue pyramid edges

    # --- Add camera axes (length = scale) ---
    axes = np.array([
        [0, 0, 0],   # origin
        [scale, 0, 0],  # x
        [0, scale, 0],  # y
        [0, 0, scale],  # z
    ])
    axes_world = (Rt @ np.hstack((axes, np.ones((4, 1)))).T).T[:, :3]

    axis_lines = [
        [0, 1],  # x-axis
        [0, 2],  # y-axis
        [0, 3],  # z-axis
    ]
    axis_colors = [
        [1, 0, 0],  # red for x
        [0, 1, 0],  # green for y
        [0, 0, 1],  # blue for z
    ]

    # --- Combine pyramid + axes ---
    all_points = np.vstack((points_world, axes_world))
    all_lines = lines + [[p[0] + len(points_world), p[1] + len(points_world)] for p in axis_lines]
    all_colors = colors + axis_colors

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)

    return line_set
