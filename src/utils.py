
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


def BaseTriangulation(kp1,kp2,mask,K1,K2,R0,t0,R,t,matches,point_cloud): 
    ref1,ref2 = [-1]*(len(kp1)),[-1]*(len(kp2))

    img1pts = [kp1[m.queryIdx].pt for m in matches][mask]
    img2pts = [kp2[m.trainIdx].pt for m in matches][mask]
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K1).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K2).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

    pts4d = cv2.triangulatePoints(np.hstack((R0,t0)),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

    for match, X in zip(matches, pts3d):
        p_idx = len(point_cloud)
        point_cloud.append([X[0], X[1], X[2], 255, 0, 0])  # white color for now
        ref1[match.queryIdx] = p_idx
        ref2[match.trainIdx] = p_idx
    return pts3d,ref1,ref2,point_cloud

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
        count = CountFrontOfBothCameras(pts3d,R,t)
        
        if count > maxfrontpts: 
            maxfrontpts = count
            bestR,bestt = R,t
    
    return bestR,bestt,maxfrontpts

def SeedPair_PoseEstimation(kp1,desc1,kp2,desc2,K1,K2,R_0,t_0,matches):
    pts1 = [kp1[m.queryIdx].pt for m in matches]
    pts2 = [kp2[m.trainIdx].pt for m in matches]
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,ransacReprojThreshold=1.0)
    E = K2.T.dot(F.dot(K1))
    R1,R2,t = ExtractCameraPoses(E)
    configSet = [None,None,None,None]
    configSet[0] = (R1,t,Triangulate2Views(pts1[mask],pts2[mask],K1,K2,R_0,t_0,R1,t))
    configSet[1] = (R1,-t,Triangulate2Views(pts1[mask],pts2[mask],K1,K2,R_0,t_0,R1,-t))
    configSet[2] = (R2,t,Triangulate2Views(pts1[mask],pts2[mask],K1,K2,R_0,t_0,R2,t))
    configSet[3] = (R2,-t,Triangulate2Views(pts1[mask],pts2[mask],K1,K2,R_0,t_0,R2,-t))

    R,t,count  = DisambiguateCameraPose(configSet)

    return R,t,mask

def ReprojectionError(img1pts,R,t,K,pts3d):
    outh = K.dot(R.dot(pts3d.T) + t )
    out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
    return np.mean(np.sqrt(np.sum((img1pts-out)**2,axis=-1)))

def Correspondence2D_3D(img_data,img,all_points3D):
    sift=cv2.SIFT_create(nfeatures=5000)
    kp_new, desc_new = sift.detectAndCompute(img,None)
    matcher = cv2.BFMatcher(crossCheck=True)
    updated_3D_pts = []
    filtered_kp = []
    for id in img_data.keys():
        _,_,ref_arr,desc_org,_ =img_data[id]
        matches = matcher.match(desc_new, desc_org)
        matches = sorted(matches, key = lambda x:x.distance)
        img_new_idx = [m.queryIdx for m in matches]
        img_new_pts = [kp_new[m.queryIdx].pt for m in matches]
        for m in matches:
            idx_3d = ref_arr[m.trainIdx]
            if idx_3d != -1 :
                updated_3D_pts.append(all_points3D[idx_3d])
                filtered_kp.append(kp_new[m.queryIdx].pt) 
    return filtered_kp, updated_3D_pts if len(updated_3D_pts)!=0 else all_points3D, desc_new,kp_new

def Triangulate2Views(img1pts,img2pts,R1,t1,K1,R2,t2,K2):
    if Rbase is None: 
        Rbase = np.eye((3,3)) 
    if tbase is None: 
        tbase = np.zeros((3,1))

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
