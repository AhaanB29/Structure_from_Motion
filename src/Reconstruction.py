from BoW import BoW_main, RANSAC, BoW
import pandas as pd
import numpy as np
import json
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
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
    t = t.reshape(3, 1)      # ensure 3×1
    Rt = np.hstack((R, t))   # 3×4
    return K @ Rt            # 3×4

def projection_2D_3D(scene_graph,all_R,all_k,all_t,img_id,path,orb):
    all_points_3D = []
    visited_edges = set() 
    for q in range(len(img_id)):
        if img_id[q] in scene_graph.keys():
            for c in scene_graph[img_id[q]]:
                if (img_id[q],c) not in visited_edges and (c,img_id[q]) not in visited_edges:
                    visited_edges.add((img_id[q],c))
                    _,num_inliers,pts_c,pts_q = RANSAC(img_id[q],c,path,orb)
                    if num_inliers < 2:
                        continue
                    c_idx = img_id.index(c,0,len(img_id))
                    P_q = projection_mat(all_k[q],all_R[q],all_t[q])
                    P_c=  projection_mat(all_k[c_idx],all_R[c_idx],all_t[c_idx])
                    points_4D = cv2.triangulatePoints(P_q,P_c,pts_q,pts_c) # (X,Y,Z,W)
                    point_3D = (points_4D[:3, :] / points_4D[3, :]).T
                    all_points_3D.append(point_3D)
    return all_points_3D

###########################################################################
if __name__ == '__main__':
    csv_path = '/media/ahaanbanerjee/Crucial X9/SfM/src/camera_params_church.csv'
    img_path = "/media/ahaanbanerjee/Crucial X9/SfM/Data/train/church/images/"
    img_id, descprs, scene_graph,orb = BoW_main()
    all_R, all_t, all_k = Camera_Params(csv_path,img_id)
    
    # print(scene_graph)
    points_3D = projection_2D_3D(scene_graph,all_R,all_k,all_t,img_id,img_path,orb)
    all_pts = np.vstack(points_3D)
    print(all_pts.shape)
# then visualize all_pts
    
    #############################
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], s=1, c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud Projection')
    plt.show()
    
