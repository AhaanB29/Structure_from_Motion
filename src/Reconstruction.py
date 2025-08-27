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
    failed_images =[]

    R_0 = np.eye(3)
    t_0 = np.zeros((3, 1))
    K1 = all_k[img_id.index(seed_pair[0])]
    K2 = all_k[img_id.index(seed_pair[1])]

    img1 = cv2.imread(path+seed_pair[0])
    img2 = cv2.imread(path+seed_pair[1])
    img1,K1,_ = resize(img1,K1)
    img2,K2,_ = resize(img2,K2)
    ################################# SEED PAIR TRIANGULATION ####################################################
    kp1,desc1,kp2,desc2,matches= GetImageMatches(img1,img2)            # returns matched Keypoints in img1 and img2
    R,t,mask= SeedPair_PoseEstimation(kp1,desc1,kp2,desc2,K1,K2,R_0,t_0,matches)  # estimates R and t for the other camera frame , returns F_matrix mask
    print(len(all_points_3D))
    # t = t/np.linalg.norm(t)
    pts_3d,pts1,pts2,matches= BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,R,t,matches)
    scale_factor = 7.38
    # pts_3d = pts_3d * scale_factor
    # t = t * 
    #pts_3d,_,_,_= bundle_adjustment(pts_3d,pts1,pts2,K1,K2,R,t)
    #pts_3d,pts1,pts2,matches= BaseTriangulation(kp1,kp2,mask,K1,K2,R_0,t_0,Rnew,tnew,matches)
    ref1,ref2= [-1]*len(kp1),[-1]*len(kp2)
    clrs = get_colors(seed_pair[0],pts1,path,K1)
    for match, X,C in zip(matches, pts_3d,clrs):
        p_idx = len(all_points_3D)
        all_points_3D.append([X[0], X[1], X[2], C[2],C[1],C[0]])
        ref2[match.trainIdx] = p_idx
        ref1[match.queryIdx] = p_idx

        
    print(pts_3d.shape)
    
    err1 = ReprojectionError(pts1,R_0,t_0,K1,pts_3d)
    err2 = ReprojectionError(pts2,R,t,K2,pts_3d)
    all_errors.append(err1)
    all_errors.append(err2)
    print(err1,err2)
    visited_ids.append(seed_pair[0])
    visited_ids.append(seed_pair[1])
    image_data[seed_pair[0]] = (R_0,t_0,K1,ref1,desc1,kp1)
    image_data[seed_pair[1]] = (R,t,K2,ref2,desc2,kp2)
    ##VISUALISATION################################
    pcd = o3d.geometry.PointCloud()
    temp = np.array(all_points_3D, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(temp[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(temp[:, 3:])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    cam_poses = [(R_0,t_0,K1),(R,t,K2)]
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
        unregistered_ids = [i for i in img_id if i not in visited_ids and i not in failed_images]
        
        if not unregistered_ids:
            print("No more images to register")
            break
        
        try:
            best_img_id, matched_2D_pts, matched_3D_pts, desc_new, kp_new, K_new = select_next_image(
                image_data, all_points_3D, unregistered_ids, path, all_k, img_id)
        except Exception as e:
            print(f"Failed to select next image: {e}")
            break
        
        print(f"------------------------------Image Registered: {best_img_id}")
        print(f"Images left to be registered: {len(unregistered_ids)-1}")
        print(f"3D-2D correspondences: {np.array(matched_3D_pts).shape} , {np.array(matched_2D_pts).shape}")
        
        # Check if we have sufficient matches for PnP
        if len(matched_3D_pts) < 8:
            print(f"Insufficient matches ({np.array(matched_3D_pts).shape}) for {best_img_id}. Adding to failed list.")
            failed_images.append(best_img_id)
            continue
        
        ###################################Estimating Pose for New Added Frame #############################
        retval = False
        inliers = None
        pnp_methods = [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_ITERATIVE]
        
        for method in pnp_methods:
            try:
                if matched_3D_pts.shape[0] < (4 if method == cv2.SOLVEPNP_P3P else 6):
                    continue
                
                retval, Rvec, tnew, inliers = cv2.solvePnPRansac(
                    matched_3D_pts, matched_2D_pts,
                    K_new, None,
                    confidence=0.99,
                    reprojectionError=4.0,
                    flags=method,
                    iterationsCount=2000
                )
                
                if retval and inliers is not None and len(inliers) >= 4:
                    method_name = {cv2.SOLVEPNP_EPNP: "EPNP", cv2.SOLVEPNP_P3P: "P3P", 
                                 cv2.SOLVEPNP_ITERATIVE: "ITERATIVE"}[method]
                    print(f"PnP successful with {method_name}: {len(inliers)} inliers")
                    break
            except cv2.error as e:
                print(f"PnP method failed: {e}")
                continue
        
        if not retval or inliers is None or len(inliers) < 4:
            print(f"All PnP methods failed for {best_img_id}. Adding to failed list.")
            failed_images.append(best_img_id)
            continue
        
        visited_ids.append(best_img_id)
        
        # Refine pose if we have enough inliers
        if len(inliers) >= 6:
            Rvec, tnew = cv2.solvePnPRefineLM(
                matched_3D_pts[inliers.flatten()], 
                matched_2D_pts[inliers.flatten()], 
                K_new, None, Rvec, tnew
            )
        
        Rnew, _ = cv2.Rodrigues(Rvec)
        
        ################################################################
        ref = [-1] * len(kp_new)
        
        ####################Best image to triangulate with############################
        best_overlap_id = None
        best_overlap_count = 0
        
        for prev_img_id in visited_ids[:-1]:  # Exclude current image
            if prev_img_id == best_img_id:
                continue
            cam_data = image_data[prev_img_id]
            if len(cam_data) == 6:  # Old format
                ROld, tOld, kOld, Ref_Old, descOld, kpOld = cam_data
            try:
                matcher = cv2.BFMatcher(crossCheck=True)
                temp_matches = matcher.match(descOld, desc_new)
                
                if len(temp_matches) > best_overlap_count:
                    best_overlap_count = len(temp_matches)
                    best_overlap_id = prev_img_id
            except Exception as e:
                print(f"Matching failed with {prev_img_id}: {e}")
                continue
        
        if best_overlap_id is None:
            if len(visited_ids) >= 2:
                best_overlap_id = visited_ids[-2]  # Fallback to previous image
            else:
                print(f"No valid overlap image found for {best_img_id}")
                failed_images.append(best_img_id)
                visited_ids.remove(best_img_id)
                continue
        
        # Get data for best overlapping image
        cam_data = image_data[best_overlap_id]
        if len(cam_data) == 6:
            ROld, tOld, kOld, Ref_Old, descOld, kpOld = cam_data

        
        print(f"Triangulating with {best_overlap_id} ({best_overlap_count} potential matches)")
        
        ##########################
        print('[Info]: Feature Matching..')
        try:
            matcher = cv2.BFMatcher(crossCheck=True)
            matches = matcher.match(descOld, desc_new)
        except Exception as e:
            print(f"Feature matching failed: {e}")
            ref = [-1] * len(kp_new)
            image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
            continue
        
        if len(matches) < 10:
            print(f"Insufficient raw matches ({len(matches)}) with {best_overlap_id}. Skipping triangulation.")
            ref = [-1] * len(kp_new)
            image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
            continue
        
        matches = sorted(matches, key=lambda x: x.distance)
        imgOldPts = np.array([kpOld[m.queryIdx].pt for m in matches])
        imgNewPts = np.array([kp_new[m.trainIdx].pt for m in matches])
        
        # Pruning the matches using fundamental matrix
        print('[Info]: Pruning the Matches..')
        try:
            F, mask = cv2.findFundamentalMat(imgOldPts, imgNewPts, cv2.FM_RANSAC, 1.0, 0.99)
        except cv2.error as e:
            print(f"Fundamental matrix computation failed: {e}")
            ref = [-1] * len(kp_new)
            image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
            continue
        
        if mask is None or mask.sum() < 8:
            print(f"Fundamental matrix yielded insufficient inliers ({mask.sum() if mask is not None else 0})")
            ref = [-1] * len(kp_new)
            image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
            continue
        
        print('[Info]: Triangulating..')
        newPts, imgOldPts, imgNewPts, valid_matches = BaseTriangulation(
            kpOld, kp_new, mask, kOld, K_new, ROld, tOld, Rnew, tnew, matches)
        #newPts,_,_,_= bundle_adjustment(newPts,imgOldPts,imgNewPts,kOld,K_new,Rnew,tnew)
        if len(newPts) == 0:
            print("Triangulation yielded no valid 3D points")
            ref = [-1] * len(kp_new)
            image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
            continue
        
        try:
            error= ReprojectionError(imgNewPts, Rnew, tnew, K_new, newPts)
            print(f"Reprojection Error pre BA: {error:.3f}")
            all_errors.append(error)
        except Exception as e:
            print(f"Error calculation failed: {e}")

        clrs = get_colors(best_img_id, imgNewPts, path, K_new)
        print(np.array(newPts).shape)
        for match, X, C in zip(valid_matches, newPts, clrs):
            p_idx = len(all_points_3D)
            all_points_3D.append([X[0], X[1], X[2], C[2], C[1], C[0]])
            if ref[match.trainIdx] == -1:
                ref[match.trainIdx] = p_idx

        
        image_data[best_img_id] = (Rnew, tnew, K_new, ref, desc_new, kp_new)
        ##################################------BA---------######################################
        optimized_points, image_data,ba_success = run_incremental_ba_every_n_images(
            image_data, all_points_3D, visited_ids, ba_interval=5, verbose=True
        )
        
        if ba_success:
            # Update our reconstruction with BA results
            all_points_3D = optimized_points
            
            # Recalculate reprojection errors after BA
            try:
                ba_errors = compute_ba_reprojection_errors(image_data, all_points_3D, visited_ids)
                current_error = ba_errors.get(best_img_id, error)
                print(f"Reprojection Error after BA: {current_error:.3f}")
                all_errors[-1] = current_error  # Update last error
                
                # Print mean error for all cameras
                if ba_errors:
                    mean_ba_error = np.mean(list(ba_errors.values()))
                    print(f"Mean reprojection error after BA: {mean_ba_error:.3f}")
                    
            except Exception as e:
                print(f"Error calculation after BA failed: {e}")
        
        # Add camera pyramid for visualization
        try:
            pyramids.append(camera_pose_vis(K_new, 800, 600, scale=0.2, R=Rnew, t=tnew))
        except:
            pass
        ##########################################################################
        # Intermediate visualization
        if(len(visited_ids)% 6== 0):
            temp = np.vstack(all_points_3D, dtype=np.float64)
            pcd.points = o3d.utility.Vector3dVector(temp[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(temp[:, 3:])
            
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="3D Intermediate", width=800, height=600, left=50, top=50)
            vis.add_geometry(pcd)
            for pyr in pyramids:
                vis.add_geometry(pyr)
            vis.run()
            vis.destroy_window()

    
    # Final statistics
    print("Final Bundle Adjustment in process.........")
    final_optimized_points, image_data,final_ba_success = run_final_ba(
        image_data, all_points_3D, visited_ids, verbose=True
    )
    
    if final_ba_success:
        all_points_3D = final_optimized_points

    print(f"\n=== Reconstruction completed Summary ===")
    print(f"Total images processed: {len(visited_ids)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Total 3D points: {len(all_points_3D)}")
    print(f"Mean reprojection error: {np.mean(all_errors):.3f}")
    
    return all_points_3D, all_errors,pyramids




###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/artefacts/camera_intrinsics_facade_ETHZ.csv'
    img_path =  "/media/ahaanbanerjee/Crucial X9/SfM/Data/facade/images/dslr_images/"
    img_id, descprs, scene_graph,pair,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    #K =np.array([ [2759.48 ,0 ,1520.69],[0 ,2764.16 ,1006.81],[0, 0, 1]],dtype=np.float32).reshape(3,3)  #Founatain
    #K =np.array([ [1520.4 ,0 ,302.32],[0 ,1525.9,246.87],[0, 0, 1]],dtype=np.float32).reshape(3,3)       #Temple
    #all_k = [K]*len(img_id)
    #pair, _,_ = select_seed_pair(img_id,img_path,all_k)
    #sift = cv2.SIFT_create(nfeatures=8000)
    #pair = ('0000.jpg', '0001.jpg')  #Fountain
    pair = ('DSC_0334.JPG', 'DSC_0335.JPG') #Facade ETHZ
    #pair = ('templeR0021.png','templeR0022.png') #Temple
    #pair = ('00008.png', '00009.png') 
    all_pts,all_errors,pyramids = SfM(pair,all_k,img_id,img_path)

    all_pts = np.vstack(all_pts,dtype=np.float32)
    print(all_pts.shape)
    # #############################
    vis = o3d.visualization.Visualizer()
    vis.create_window("Structure from Motion", width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(all_pts[:,3:])  
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # Visualize
    vis.add_geometry(pcd)
    for pyr in pyramids:
                vis.add_geometry(pyr)
    vis.run()
    vis.destroy_window() 
    Toply(all_pts[:,:3],all_pts[:,3:],filename='temple_pcd.ply')
