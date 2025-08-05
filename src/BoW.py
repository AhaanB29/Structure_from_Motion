import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import pickle
######################################## Creation BoW #############################
class BoW:
    def __init__(self,branching_factor=10,max_depth = 3):
        self.max_depth = max_depth
        self.bf = branching_factor
        self.tree = None
        self.leaf_count = 0
    
    class Node:
        def __init__(self,centroid=None,depth=0, children=None):
            self.depth = depth
            self.centroid = centroid
            self.children = children if children is not None else []
            self.is_leaf = False
            self.img_id_lst=[]
    def fit(self,descriptors):
        self.leaf_count=0
        self.tree = self.recursive_clustering(descriptors,0)

    def recursive_clustering(self,descriptors,depth):
        if((depth == self.max_depth) or (len(descriptors)<self.bf)):
           node = self.Node(centroid = np.mean(descriptors,axis=0),depth=depth,children=None)
           node.is_leaf = True
           self.leaf_count+=1
           return node
        
        k = min(self.bf,len(descriptors))
        clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clustering.fit_predict(descriptors)

        node = self.Node(children = [], depth = depth)
        node.centroid = clustering.cluster_centers_
        for c_id in range(k):
            cluster_desc = descriptors[labels==c_id] # this is to get the subset of the descp in that particular cluster ID
            if(len(cluster_desc) >0):
                child = self.recursive_clustering(cluster_desc,depth+1)
                node.children.append(child)
        return node

    def find_leaf(self,descriptor):
        node= self.tree
        while not node.is_leaf and node.children:
            # calculate the diff with all the centroids
            centroids = node.centroid
            values = np.linalg.norm(centroids - descriptor, axis=1)
            idx = np.argmin(values)
            node = node.children[idx]
        return node
    
    def inv_index(self, image_name, descriptors):
        """Assign image_name to all leaves matching its descriptors."""
        for d in descriptors:
            leaf = self.find_leaf(d)
            if image_name not in leaf.img_id_lst:
                leaf.img_id_lst.append(image_name)

    def query(self, query_descriptors):
        """Return filenames ranked by vote count for query descriptors."""
        votes = {}
        for d in query_descriptors:
            leaf = self.find_leaf(d)
            for img_name in leaf.img_id_lst:
                if img_name not in votes.keys():
                    votes[img_name] = 1
                else :
                    votes[img_name]+=1
        # sort by descending votes
        return sorted(votes.items(), key=lambda x: x[1], reverse=True)
        #return votes
##############################################################################
    
def feature_extractor(dir_path,orb):
    all_descriptors = []
    # Use Feature Extractor to Key key points and descriptors
    img_id=[]
    for img_name in os.listdir(dir_path):
        img = cv2.imread(dir_path+img_name,cv2.IMREAD_GRAYSCALE)
        kp,descriptors = orb.detectAndCompute(img,None)
        all_descriptors.append(descriptors)
        img_id.append(img_name)
    return img_id,np.vstack(all_descriptors)


def scene_graph(img_id,descriptors,tree):
    graph = defaultdict(list)
    for id,desc in zip(img_id,descriptors):
        votes = tree.query(desc)
        for other,freq in votes:
            if id!=other:
                score = freq/len(desc)
                if(score >= 0.0950):
                    graph[id].append(other)
    return graph

def RANSAC(query_img,candidate_img,path,orb):
    q_img = cv2.imread(path+query_img,cv2.IMREAD_GRAYSCALE)
    c_img = cv2.imread(path+candidate_img,cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=8000)
    kp_q,descriptors_q = sift.detectAndCompute(q_img,None)
    kp_c,descriptors_c = sift.detectAndCompute(c_img,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # Increased number of checks for better matching

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_q, descriptors_c, k=2)

    good = [] #sorted(matches,key=lambda x:x.distance)
    for m,n in matches:
        if (m.distance < 0.7*n.distance):
            good.append(m) 
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    pts_q= np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    
    # F, mask = cv2.findFundamentalMat(
    #     pts_c, pts_q, cv2.FM_RANSAC,
    #     ransacReprojThreshold=1.0, confidence=0.99
    # )

    # if mask is None:
    #     return [], 0, kp_q, kp_c

    # inlier_matches = [good[i] for i in range(len(good)) if mask[i][0]]
    # num_inliers = np.sum(mask)
    # pts_c = np.float32([kp_c[m.trainIdx].pt for m in inlier_matches]).T.reshape(2, -1)
    # pts_q = np.float32([kp_q[m.queryIdx].pt for m in inlier_matches]).T.reshape(2, -1)
    return good, len(good), pts_c, pts_q



def RANSAC_refinement_Graph(graph,path,orb):
    verified_graph = defaultdict(list)
    seed_pair = (None,None)
    max_inlier = -1
    for query_img, candidates in graph.items():
        for candidate_img in candidates:
            _,num_inliers,_,_ = RANSAC(query_img, candidate_img,path,orb)
            if num_inliers >= 29:  # e.g., 15-30
                verified_graph[query_img].append((candidate_img,num_inliers))
                if(num_inliers>max_inlier):
                    seed_pair = (query_img,candidate_img)
                    max_inlier= num_inliers
    return verified_graph,seed_pair,max_inlier
###############################################################
def BoW_main():
    path = "/media/ahaanbanerjee/Crucial X9/SfM/Data/IMC/train/church/images/"
    orb = cv2.ORB_create(nfeatures=5000)
    img_id , descprs = feature_extractor(path,orb)
    flattned_desc = descprs.reshape(-1,32)
    bow = BoW(branching_factor=15,max_depth=8)
    # bow.fit(flattned_desc)
    # print(bow.leaf_count)
    # #print(bow.tree.centroid)
    # for img_name, descs in zip(img_id, descprs):
    #     bow.inv_index(img_name, descs)

    # with open('hkm_tree_late.pkl', 'wb') as f:
    #     pickle.dump(bow, f)

    print("TreeDone")
# --- Later, to load it back ---
    with open('hkm_tree_late.pkl', 'rb') as f:
        loaded_tree = pickle.load(f)
    
    # sc_grph = scene_graph(img_id,descprs,loaded_tree)
    # print(sc_grph)
    # #print('#####')
    
    # # refined_grph,pair,inlier =RANSAC_refinement_Graph(sc_grph,path,orb)
    # with open('scence_grph_org.pkl', 'wb') as f:
    #     pickle.dump(sc_grph, f)

    with open('scence_grph.pkl', 'rb') as f:
        graph = pickle.load(f)
    pair = ('00069.png', '00066.png') 
    return img_id,descprs,graph,pair,orb

if __name__ == '__main__':
    BoW_main()