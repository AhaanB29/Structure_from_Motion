from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from BA import bundle_adjustment
from utils import *


def SfM(scene_graph, seed_pair, all_k, img_id, path, detector):
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

    pts_3d,ref1,ref2,all_points_3D = BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,R,t,matches,all_points_3D)
    img1pts = [kp1[m.queryIdx].pt for m in matches]
    img2pts = [kp2[m.trainIdx].pt for m in matches]

    err1 = ReprojectionError(img1pts[mask],R_0,t_0,K1,pts_3d)
    err2 = ReprojectionError(img2pts[mask],R,t,K2,pts_3d)
    all_errors.append(err1)
    all_errors.append(err2)
    visited_ids.add(seed_pair[0])
    visited_ids.add(seed_pair[1])

    image_data[seed_pair[0]] = (R_0,t_0,K1,ref1,desc1,kp1)
    image_data[seed_pair[1]] = (R,t,K2,ref2,desc2,kp2)
    for i in range(len(img_id)):
        if img_id[i] not in visited_ids:
            visited_ids.append(img_id[i])
            K_new= all_k[img_id.index(img_id[i])]
            img_new = cv2.imread(path+img_id[i])
            matched_2D_pts, matched_3D_pts, desc_new,kp_new= Correspondence2D_3D(image_data,K_new,img_new)
            retval,Rvec,tnew,mask3gt = cv2.solvePnPRansac(matched_3D_pts[:,np.newaxis],matched_2D_pts[:,np.newaxis],
                                            K_new,None,confidence=.99,flags=cv2.SOLVEPNP_DLS)
            Rnew,_=cv2.Rodrigues(Rvec)
            for (ROld, tOld, kOld,_,descOld,kpOld) in image_data.values(): 
                #Matching between old view and newly registered view.. 
                print ('[Info]: Feature Matching..')
                matcher = cv2.BFMatcher(crossCheck=True)
                matches = matcher.match(descOld, desc_new)
                matches = sorted(matches, key = lambda x:x.distance)
                imgOldPts = [kpOld[m.queryIdx].pt for m in matches]
                imgNewPts = [kp_new[m.queryIdx].pt for m in matches]
                #Pruning the matches using fundamental matrix..
                print ('[Info]: Pruning the Matches..')
                F,mask=cv2.findFundamentalMat(imgOldPts,imgNewPts,cv2.FM_RANSAC,1.0,0.99)
                mask = mask.flatten().astype(bool)
                imgOldPts=imgOldPts[mask]
                imgNewPts=imgNewPts[mask]
                #Triangulating new points
                print ('[Info]: Triangulating..')
                newPts = Triangulate2Views(imgOldPts,imgNewPts,ROld,tOld,kOld,Rnew,tnew,K_new)

            if len(all_points_3D) == 0:
                continue


            #points_3D, R_total, t_total = bundle_adjustment(points_3D.T, src_pts, dest_pts, K1, K2, R_total, t_total)
            #current_clrs = np.vstack(all_colors)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(current_pts)
            # #pcd.colors = o3d.utility.Vector3dVector(current_clrs)  

            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

            # # Visualize
            # o3d.visualization.draw_geometries(
            #     [pcd],  window_name="3D Intermediate",
            #     width=800,
            #     height=600,
            #     left=50,
            #     top=50
            # )
            # print('Mean depth:', np.mean(current_pts[:, 2]),
            #       'Min:', np.min(current_pts[:, 2]),
            #       'Max:', np.max(current_pts[:, 2]))

    return all_points_3D, all_colors



###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/artefacts/camera_params_church.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC/train/church/images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    print(pair)
    sift = cv2.SIFT_create(nfeatures=8000)
  
    all_pts,all_clrs = SfM(scene_graph,pair,all_R,all_k,all_t,img_id,img_path,sift)

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
