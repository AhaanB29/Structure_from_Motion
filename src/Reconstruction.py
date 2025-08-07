from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from BA import bundle_adjustment

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


def projection_mat(K, R, t):
    # = t/np.linalg.norm(t)
    #print(t.shape)
    t = t.reshape(3, 1)      
    Rt = np.hstack((R, t))   
    return K @ Rt            

def triangulation_F(q_img, c_img, path, detector, img_id, all_k, R_previous, t_previous):
    _, num_inliers, pts_c, pts_q = RANSAC(q_img, c_img, path, detector)
    q_idx = img_id.index(q_img)
    c_idx = img_id.index(c_img)
    K1 = all_k[q_idx]
    K2 = all_k[c_idx]
   
    if num_inliers < 8 or pts_q.shape[0] < 8:
        print(num_inliers)
        print(q_img,c_img)
        return '','','','','','',''

    F, mask = cv2.findFundamentalMat(pts_q, pts_c, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or F.shape != (3,3):
        return '','','','','','','' 
   
    # Compute Essential Matrix
    E = K2.T @ F @ K1

    # Normalize image points
    pts_q_norm = cv2.undistortPoints(pts_q.reshape(-1, 1, 2), K1, None)
    pts_c_norm = cv2.undistortPoints(pts_c.reshape(-1, 1, 2), K2, None)

    # Recover pose using normalized coordinates and identity matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts_q_norm, pts_c_norm, np.eye(3))
    t = t / np.linalg.norm(t)
    # Compose pose
    R1_global = R_previous
    t1_global = t_previous  
    R2_global = R_previous @ R  
    t2_global = t_previous + R_previous @ t  
    
    P_q = projection_mat(K1, R1_global, t1_global)
    P_c = projection_mat(K2, R2_global, t2_global)
    # Projection matrices (undistorted points used for pose, but we project using original intrinsics)
    # P_q = projection_mat(K1, R, t)
    # P_c = projection_mat(K2, R_total, t_total)

    # Triangulate using original pixel coordinates
    
    points_4D = cv2.triangulatePoints(P_q, P_c, pts_q, pts_c)
    points_3D = (points_4D[:3, :] / points_4D[3, :]).T

    return points_3D, pts_q, pts_c, R2_global, t2_global, K1, K2


def triangulation(q_img,c_img,path,detector,img_id,all_k,all_R,all_t):
    match, num_inliers, pts_c, pts_q = RANSAC(q_img, c_img, path, detector)

    if num_inliers < 8 or pts_q.shape[0] < 8:
        print(num_inliers)
        print(q_img,c_img)
        return '','','','','','',""

    try:
        q_idx = img_id.index(q_img)
        c_idx = img_id.index(c_img)
    except ValueError:
        return
    
    P_q = projection_mat(all_k[q_idx], all_R[q_idx], all_t[q_idx])
    P_c = projection_mat(all_k[c_idx], all_R[c_idx], all_t[c_idx])
    points_4D = cv2.triangulatePoints(P_q, P_c, pts_q, pts_c)
    point_3D = (points_4D[:3, :] / points_4D[3, :]).T
    return point_3D,pts_q,pts_c,all_R[c_idx],all_t[c_idx],all_k[q_idx],all_k[c_idx]


def projection_2D_3D(scene_graph, seed_pair, all_R, all_k, all_t, img_id, path, detector):
    all_points_3D = []
    all_colors = []
    visited_pairs = set()

    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    R_previous = R_total.copy()
    t_previous = t_total.copy()

    #Initial triangulation with seed pair
    points_3D, src_pts, dest_pts, R_total, t_total, K1, K2 = triangulation_F(
        seed_pair[0], seed_pair[1], path, detector, img_id, all_k, R_previous,t_previous)
    points_3D, R_total, t_total = bundle_adjustment(points_3D.T, src_pts, dest_pts, K1, K2, R_total, t_total)

    # Filter by Z (depth) between 0 and 200
    z_vals = points_3D[:, 2]
    valid_mask = (
    np.isfinite(points_3D).all(axis=1) &  # remqove NaN or inf
    (z_vals > 0) &                        # in front of the camera
    (z_vals < 25) )                      # depth range filter)  # avoid "infinity" points

    filtered_pts = points_3D[valid_mask]
    #filtered_src_pts = src_pts.reshape(-1, 2)[mask]
    #filtered_colors = get_colors(seed_pair[0], src_pts, path)

    all_points_3D.append(filtered_pts)
    #all_colors.append(filtered_colors)
    visited_pairs.add(seed_pair)

    R_total = R_previous.copy()
    t_total = t_previous.copy()

    for i in sorted(scene_graph.keys()):
        for a in scene_graph[i]:
            pair = (i, a[0])
            if pair not in visited_pairs and (a[0],i) not in visited_pairs:
                visited_pairs.add(pair)

                points_3D, src_pts, dest_pts, R_total, t_total, K1, K2 = triangulation_F(i, a[0],path, detector, img_id, all_k,R_total, t_total)
                if len(points_3D) == 0:
                    continue

                points_3D, R_total, t_total = bundle_adjustment(points_3D.T, src_pts, dest_pts, K1, K2, R_total, t_total)

                # Filter by Z (depth)
                z_vals = points_3D[:, 2]
                valid_mask = (np.isfinite(points_3D).all(axis=1) & (z_vals>0)&  (z_vals < 35))
                #mask = (z_vals >= 0) & (z_vals < 30)
                filtered_pts = points_3D[valid_mask]
                #filtered_src_pts = src_pts.reshape(-1, 2)[mask]
                filtered_colors = get_colors(i, src_pts, path)

                all_points_3D.append(filtered_pts)
                #all_colors.append(filtered_colors)

                R_previous = R_total.copy()
                t_previous = t_total.copy()
                # Visualization
                current_pts = np.vstack(all_points_3D)
                #current_clrs = np.vstack(all_colors)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(current_pts)
                #pcd.colors = o3d.utility.Vector3dVector(current_clrs)  

                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

                # Visualize
                o3d.visualization.draw_geometries(
                    [pcd],  window_name="3D Intermediate",
                    width=800,
                    height=600,
                    left=50,
                    top=50
                )
                print('Mean depth:', np.mean(current_pts[:, 2]),
                      'Min:', np.min(current_pts[:, 2]),
                      'Max:', np.max(current_pts[:, 2]))

    return all_points_3D, all_colors



###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/artefacts/camera_params_church.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC/train/church/images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    print(pair)
    sift = cv2.SIFT_create(nfeatures=8000)
  
    all_pts,all_clrs = projection_2D_3D(scene_graph,pair,all_R,all_k,all_t,img_id,img_path,sift)

    all_pts = np.vstack(all_pts)
    #all_clrs = np.vstack(all_clrs,dtype=np.float32)
    print(all_pts.shape)
    # #############################
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    #pcd.colors = o3d.utility.Vector3dVector(all_clrs)  

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],  window_name="3D Reconstruction with Color",
        width=800,
        height=600,
        left=50,
        top=50
    )
