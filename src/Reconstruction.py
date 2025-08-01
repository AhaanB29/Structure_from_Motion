from BoW import BoW_main, RANSAC_refinement_Graph, feature_extractor, BoW
import pandas as pd
import numpy as np
import json

def Camera_Params(file_path, img_id):
    data = pd.read_csv(file_path)

    # Extract corresponding matrices for each image ID
    all_R, all_t, all_k = [], [], []
    for i in img_id:
        R =np.array((data[data['image_name'] == i]['rotation_matrix']).values[0].split(';'),dtype=np.float32).reshape(3,3)
        t = np.array((data[data['image_name']== i]['translation_vector']).values[0].split(';'),dtype=np.float32).reshape(1,3)
        K = np.array((data[data['image_name']== i]['calibration_matrix']).values[0].split(';'),dtype=np.float32).reshape(3,3)

        all_R.append(R)
        all_t.append(t)
        all_k.append(K)
    return all_R, all_t, all_k

if __name__ == '__main__':
    path = '/media/ahaanbanerjee/Crucial X9/SfM/src/camera_params_church.csv'
    img_id_2, descprs, scene_graph = BoW_main()
    all_R, all_t, all_k = Camera_Params(path,img_id_2)
    print(type(all_R[0]))
