from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from BA import bundle_adjustment_local
from utils import *
from initialisation import select_seed_pair

def SfM(seed_pair, all_k, img_id, path):
    all_points_3D = []
    all_colors = []
    all_errors = []
    visited_ids = []
    image_data = {}
    R_0 = np.eye(3)
    t_0 = np.zeros((3, 1))
    K1 = all_k[img_id.index(seed_pair[0])]
    K2 = all_k[img_id.index(seed_pair[1])]
    img1 = cv2.imread(path+seed_pair[0])
    img2 = cv2.imread(path+seed_pair[1])


    kp1,desc1,kp2,desc2,matches= GetImageMatches(img1,img2)            # returns matched Keypoints in img1 and img2
    R,t,mask= SeedPair_PoseEstimation(kp1,desc1,kp2,desc2,K1,K2,R_0,t_0,matches)  # estimates R and t for the other camera frame , returns F_matrix mask
    print(len(all_points_3D))
    pts_3d,ref1,ref2,all_points_3D = BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,R,t,matches,all_points_3D)
    print(pts_3d.shape)
    img1pts = np.array([kp1[m.queryIdx].pt for m in matches])
    img2pts = np.array([kp2[m.trainIdx].pt for m in matches])

    print(np.array(all_points_3D).shape)
    err1 = ReprojectionError(img1pts[mask],R_0,t_0,K1,pts_3d)
    err2 = ReprojectionError(img2pts[mask],R,t,K2,pts_3d)
    all_errors.append(err1)
    all_errors.append(err2)
    visited_ids.append(seed_pair[0])
    visited_ids.append(seed_pair[1])

    image_data[seed_pair[0]] = (R_0,t_0,K1,ref1,desc1,kp1)
    image_data[seed_pair[1]] = (R,t,K2,ref2,desc2,kp2)
    while len(visited_ids) < len(img_id):
        #----------------------------------------Registration--------------------------------------------#
        unregistered_ids = [i for i in img_id if i not in visited_ids]
        best_img_id,matched_2D_pts, matched_3D_pts, desc_new,kp_new = select_next_image(image_data, all_points_3D, unregistered_ids, path)
        visited_ids.append(best_img_id)
        K_new= all_k[img_id.index(best_img_id)]
        img_new = cv2.imread(path+best_img_id)
        print(best_img_id)
        #print(matched_3D_pts.shape,matched_2D_pts.shape)
        if matched_3D_pts.shape[0] < 6:
            continue
        retval,Rvec,tnew,mask3gt = cv2.solvePnPRansac(matched_3D_pts[:,np.newaxis],matched_2D_pts[:,np.newaxis],
                                        K_new,None,confidence=.99,flags=cv2.SOLVEPNP_DLS)
        Rnew,_=cv2.Rodrigues(Rvec)
        ref = [-1]*len(kp_new)

        # Triangulation
        for (ROld, tOld, kOld,Ref_Old,descOld,kpOld) in image_data.values(): 
    
            print ('[Info]: Feature Matching..')
            matcher = cv2.BFMatcher(crossCheck=True)
            matches = matcher.match(descOld, desc_new)
            matches = sorted(matches, key = lambda x:x.distance)
            imgOldPts = np.array([kpOld[m.queryIdx].pt for m in matches])
            imgNewPts = np.array([kp_new[m.trainIdx].pt for m in matches])

            #Pruning the matches using fundamental matrix..
            print ('[Info]: Pruning the Matches..')
            F,mask=cv2.findFundamentalMat(imgOldPts,imgNewPts,cv2.FM_RANSAC,1.0,0.99)
            mask = mask.flatten().astype(bool)
            imgOldPts=imgOldPts[mask]
            imgNewPts=imgNewPts[mask]
            #Triangulating new points
            print ('[Info]: Triangulating..')
            newPts = Triangulate2Views(imgOldPts,imgNewPts,ROld,tOld.reshape((3,1)),kOld,Rnew,tnew.reshape((3,1)),K_new)
            print(newPts.shape)
            mask_finite = np.isfinite(newPts).all(axis=1)
            # Cheirality mask
            pts_cam1 = (ROld @ newPts.T + tOld.reshape(3,1)).T
            pts_cam2 = (Rnew @ newPts.T + tnew.reshape(3,1)).T
            mask_front = (pts_cam1[:,2] > -0.0001) & (pts_cam2[:,2] > -0.00001)

            # Depth mask (limit to 20 units in first camera frame)
            mask_depth = (pts_cam1[:,2] < 50) & (pts_cam2[:,2] < 50)

            #Angle Mask
            view_dir1 = pts_cam1 / np.linalg.norm(pts_cam1, axis=1)[:, None]
            view_dir2 = pts_cam2 / np.linalg.norm(pts_cam2, axis=1)[:, None]
            angles = np.arccos(np.clip(np.sum(view_dir1 * view_dir2, axis=1), -1, 1))
            mask_angle = np.degrees(angles) > 1.5  # COLMAP default
            # Combine all masks
            mask_all =  mask_depth & mask_finite & mask_front & mask_angle

            # Apply once
            newPts = newPts[mask_all]
            imgOldPts = imgOldPts[mask_all]
            imgNewPts = imgNewPts[mask_all]

            error = ReprojectionError(imgNewPts,Rnew,tnew,K_new,newPts)
            print("Reprojection Error: ", error)
            all_errors.append(error)
            print("After all Pruning: ",newPts.shape)
            if len(newPts) > 0:
                for match, X in zip(matches, newPts):
                    p_idx = len(all_points_3D)
                    all_points_3D.append([X[0], X[1], X[2], 255, 0, 0])  # white color for now
                    ref[match.trainIdx] = p_idx

                
        image_data[best_img_id] = (Rnew,tnew,K_new,ref,desc_new,kp_new)
        


    #points_3D, R_total, t_total = bundle_adjustment(points_3D.T, src_pts, dest_pts, K1, K2, R_total, t_total)
    #current_clrs = np.vstack(all_colors)
        pcd = o3d.geometry.PointCloud()
        temp= np.array(all_points_3D, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(temp[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(temp[:,3:])  

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
        # print('Mean depth:', np.mean(current_pts[:, 2]),
        #       'Min:', np.min(current_pts[:, 2]),
        #       'Max:', np.max(current_pts[:, 2]))

    return all_points_3D,all_errors



###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/artefacts/camera_params_church.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC/train/church/images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    #pair, _,_, _ = select_seed_pair(img_path,img_id,all_k,)
    #sift = cv2.SIFT_create(nfeatures=8000)
  
    all_pts,all_errors = SfM(pair,all_k,img_id,img_path)

    all_pts = np.vstack(all_pts,dtype=np.float32)
    #all_clrs = np.vstack(all_clrs,dtype=np.float32)
    print(all_pts.shape)
    # #############################
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(all_pts[:,3:])  

    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],  window_name="3D Reconstruction with Color",
        width=800,
        height=600,
        left=50,
        top=50
    )
    Toply(all_pts[:,:3],filename='Church2.ply')
