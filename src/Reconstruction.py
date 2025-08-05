from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from BundleAdjustment import bundle_adjustment

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
    #t = t / np.linalg.norm(t)
    t = t.reshape(3, 1)      
    Rt = np.hstack((R, t))   
    return K @ Rt            

def triangulation(q_img,c_img,path,orb,img_id,all_k,all_R,all_t):
    _, num_inliers, pts_c, pts_q = RANSAC(q_img, c_img, path, orb)

    if num_inliers < 8 or pts_q.shape[0] < 8:
        print(num_inliers)
        print(q_img,c_img)
        return '','','','','','','','',''

    try:
        q_idx = img_id.index(q_img)
        c_idx = img_id.index(c_img)
    except ValueError:
        return
    
    P_q = projection_mat(all_k[q_idx], all_R[q_idx], all_t[q_idx])
    P_c = projection_mat(all_k[c_idx], all_R[c_idx], all_t[c_idx])
    points_4D = cv2.triangulatePoints(P_q, P_c, pts_q, pts_c)
    point_3D = (points_4D[:3, :] / points_4D[3, :]).T
    return point_3D,pts_q,pts_c,all_k[q_idx],all_R[q_idx],all_t[q_idx],all_k[c_idx],all_R[c_idx],all_t[c_idx]


def projection_2D_3D(scene_graph, seed_pair, all_R, all_k, all_t, img_id, path, orb):
    all_points_3D = []
    all_colors = []
    visited_pairs = set()
    points_3D, src_pts,dest_pts,K1,R1,t1,K2,R2,t2 = triangulation(seed_pair[0],seed_pair[1],path,orb,img_id,all_k,all_R,all_t)
    print(src_pts.shape,dest_pts.shape)
    refined_3D,refined_rv,refined_t = bundle_adjustment(points_3D.T, src_pts,dest_pts,K1,R1,t1,K2,R2,t2)
    all_points_3D.append(refined_3D)
    visited_pairs.add(seed_pair)
    all_colors.append(get_colors(seed_pair[0],src_pts,path))

    for i in range(0,len(img_id)-1):
        if(img_id[i],img_id[i+1]) not in visited_pairs:
            visited_pairs.add((img_id[i],img_id[i+1]))
            points_3D, src_pts,dest_pts,K1,R1,t1,K2,R2,t2 = triangulation(img_id[i],img_id[i+1],path,orb,img_id,all_k,all_R,all_t)
            if(len(points_3D)==0):
                continue
            # print(points_3D.shape)
            # print(src_pts.shape)
            # print(dest_pts.shape)
            all_colors.append(get_colors(img_id[i],src_pts,path))
            refined_3D,refined_rv,refined_t = bundle_adjustment(points_3D.T, src_pts,dest_pts,K1,R1,t1,K2,R2,t2)
            all_points_3D.append(refined_3D)

    return all_points_3D,all_colors


###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/camera_params_church.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC/train/church/images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    print(pair)
    # print(img_id[img_id.index('00006.png',0,len(img_id))])
    # print(all_R[5])
    # print(all_t[5])
    # print(all_k[5])
    # K_mean = np.mean(all_k, axis=0)
    # print(K_mean)
    # # print(scene_graph)
    all_pts,all_clrs = projection_2D_3D(scene_graph,pair,all_R,all_k,all_t,img_id,img_path,orb)
    # mask = np.all(np.abs(all_pts) < 5000, axis=1)
    # all_pts = all_pts[mask]
    all_pts = np.vstack(all_pts)
    all_clrs = np.vstack(all_clrs,dtype=np.float32)
    print(all_pts.shape)
    # #############################
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_clrs)  

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
