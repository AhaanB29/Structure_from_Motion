from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
#from BA import bundle_adjustment_local
from BundleAdjustment import *
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
    ################################# SEED PAIR TRIANGULATION ####################################################
    kp1,desc1,kp2,desc2,matches= GetImageMatches(img1,img2)            # returns matched Keypoints in img1 and img2
    R,t,mask= SeedPair_PoseEstimation(kp1,desc1,kp2,desc2,K1,K2,R_0,t_0,matches)  # estimates R and t for the other camera frame , returns F_matrix mask
    print(len(all_points_3D))
    #t = t/np.linalg.norm(t)
    pts_3d,pts1,pts2,matches= BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,R,t,matches)
    scale_factor = 7.38
    # pts_3d = pts_3d * scale_factor
    # t = t * 
    pts_3d,Rnew,tnew,_= bundle_adjustment(pts_3d,pts1,pts2,K1,K2,R,t)
    pts_3d,pts1,pts2,matches= BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,Rnew,tnew,matches)
    ref1,ref2= [-1]*len(kp1),[-1]*len(kp2)
    clrs = get_colors(seed_pair[0],pts1,path)
    for match, X,C in zip(matches, pts_3d,clrs):
        p_idx = len(all_points_3D)
        all_points_3D.append([X[0], X[1], X[2], C[2],C[1],C[0]])
        ref2[match.trainIdx] = p_idx
        ref1[match.queryIdx] = p_idx

        
    print(pts_3d.shape)
    
    err1 = ReprojectionError(pts1,R_0,t_0,K1,pts_3d)
    err2 = ReprojectionError(pts2,Rnew,tnew,K2,pts_3d)
    all_errors.append(err1)
    all_errors.append(err2)
    print(err1,err2)
    visited_ids.append(seed_pair[0])
    visited_ids.append(seed_pair[1])
    image_data[seed_pair[0]] = (R_0,t_0,K1,ref1,desc1,kp1)
    image_data[seed_pair[1]] = (Rnew,tnew,K2,ref2,desc2,kp2)
    ##VISUALISATION################################
    pcd = o3d.geometry.PointCloud()
    temp = np.array(all_points_3D, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(temp[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(temp[:, 3:])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    cam_poses = [(R_0,t_0,K1),(Rnew,tnew,K2)]
    pyramids = [camera_pose_vis(K, 800, 600, scale=0.2, R=R, t=t) for R, t,K in cam_poses]

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D with Camera Pyramids", width=800, height=600)
    vis.add_geometry(pcd)
    for pyr in pyramids:
        vis.add_geometry(pyr)
    vis.run()
    vis.destroy_window() 
    ###############################################################
    while len(visited_ids) < len(img_id):
        #----------------------------------------Registration--------------------------------------------#
        unregistered_ids = [i for i in img_id if i not in visited_ids]
        best_img_id,matched_2D_pts, matched_3D_pts, desc_new,kp_new = select_next_image(image_data, all_points_3D, unregistered_ids, path)
        visited_ids.append(best_img_id)
        K_new= all_k[img_id.index(best_img_id)]
        #img_new = cv2.imread(path+best_img_id)
        print("------------------------------Image Registered :------------------",best_img_id)
        print(matched_3D_pts.shape,matched_2D_pts.shape)
        if matched_3D_pts.shape[0] < 6:
            continue
        ###################################Estimating Pose for New Added Frame #############################
        retval,Rvec,tnew,inliers = cv2.solvePnPRansac(matched_3D_pts,matched_2D_pts,
                                        K_new,None,confidence=0.999,reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
        if inliers is not None:
            Rvec,tnew = cv2.solvePnPRefineLM(matched_3D_pts[inliers.flatten()],matched_2D_pts[inliers.flatten()],K_new,None,Rvec,tnew)
        # tnew = tnew/np.linalg.norm(tnew)
        Rnew,_=cv2.Rodrigues(Rvec)
        #################################################################       
        ################################################################
        ref = [-1]*len(kp_new)
        # Triangulation
        ROld, tOld, kOld,Ref_Old,descOld,kpOld = image_data[visited_ids[-2]]
        print ('[Info]: Feature Matching..')
        matcher = cv2.BFMatcher(crossCheck=True)
        matches = matcher.match(descOld, desc_new)
        matches = sorted(matches, key = lambda x:x.distance)
        imgOldPts = np.array([kpOld[m.queryIdx].pt for m in matches])
        imgNewPts = np.array([kp_new[m.trainIdx].pt for m in matches])
        
        #Pruning the matches using fundamental matrix..
        print ('[Info]: Pruning the Matches..')
        #F,mask=cv2.findFundamentalMat(imgOldPts,imgNewPts,cv2.FM_RANSAC,1.0,0.99)
        #Triangulating new points
        print ('[Info]: Triangulating..')
        newPts,imgOldPts,imgNewPts,valid_matches= BaseTriangulation(kpOld,kp_new,mask,kOld,K_new,ROld,tOld,Rnew,tnew,matches)
        error = ReprojectionError(imgNewPts,Rnew,tnew,K_new,newPts)
        print("Reprojection Error pre BA: ", error)
        # newPts=newPts*scale_factor
        ############# Bundle Adjustment############################
        if len(newPts) == 0:
                continue
        print("Bundle Adjustment in process ..............")
        newPts,Rnew,tnew,_= bundle_adjustment(newPts,imgOldPts,imgNewPts,kOld,K_new,Rnew,tnew)
        #################### Re-Triangulation##################################
        newPts,imgOldPts,imgNewPts,valid_matches= BaseTriangulation(kpOld,kp_new,mask,kOld,K_new,ROld,tOld,Rnew,tnew,matches)
        error = ReprojectionError(imgNewPts,Rnew,tnew,K_new,newPts)
        print("Reprojection Error post BA: ", error)
        all_errors.append(error)
        clrs = get_colors(best_img_id,imgNewPts,path)
        print("After all Pruning: ",newPts.shape)
        for match, X , C in zip(valid_matches, newPts,clrs):
            p_idx = len(all_points_3D)
            all_points_3D.append([X[0], X[1], X[2], C[2],C[1],C[0]])  
            if(ref[match.trainIdx] == -1):
                ref[match.trainIdx] = p_idx

        image_data[best_img_id] = (Rnew,tnew,K_new,ref,desc_new,kp_new)
        pyramids.append(camera_pose_vis(K_new, 800, 600, scale=0.2, R=Rnew, t=tnew))

    #points_3D, R_total, t_total = bundle_adjustment(points_3D.T, src_pts, dest_pts, K1, K2, R_total, t_total)
    #current_clrs = np.vstack(all_colors)
        temp = np.array(all_points_3D, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(temp[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(temp[:, 3:])

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Intermediate", width=800, height=600, left=50, top=50)
        vis.add_geometry(pcd)
        for pyr in pyramids:
            vis.add_geometry(pyr)
        vis.run()              # Shows the window; user closes it
        vis.destroy_window() 
        # print('Mean depth:', np.mean(current_pts[:, 2]),
        #       'Min:', np.min(current_pts[:, 2]),
        #       'Max:', np.max(current_pts[:, 2]))

    return all_points_3D,all_errors



###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC_G/train/brandenburg_gate/calibration.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC_G/train/brandenburg_gate/images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    #K =np.array([ [2759.48 ,0 ,1520.69],[0 ,2764.16 ,1006.81],[0, 0, 1]],dtype=np.float32).reshape(3,3)
    #K =np.array([ [3414.66, 0.0, 3036.64],[0.0, 3413.37, 2015.48],[0, 0, 1]],dtype=np.float32).reshape(3,3) #Facade
    #all_k = [K]*len(img_id)
    pair, _,_, _ = select_seed_pair(img_id,img_path,all_k)
    #sift = cv2.SIFT_create(nfeatures=8000)
    #pair = ('0000.jpg', '0001.jpg')
    #pair = ('DSC_0391.JPG', 'DSC_0392.JPG')
    # all_pts,all_errors = SfM(pair,all_k,img_id,img_path)

    # all_pts = np.vstack(all_pts,dtype=np.float32)
    # #all_clrs = np.vstack(all_clrs,dtype=np.float32)
    # print(all_pts.shape)
    # # #############################
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_pts[:,:3])
    # pcd.colors = o3d.utility.Vector3dVector(all_pts[:,3:])  

    # # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # # pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # # Visualize
    # o3d.visualization.draw_geometries(
    #     [pcd],  window_name="3D Reconstruction with Color",
    #     width=800,
    #     height=600,
    #     left=50,
    #     top=50
    # )
    # Toply(all_pts[:,:3],all_pts[:,3:],filename='Facade_norm.ply')
