# Incremental Structure from Motion (SfM)

This project is a Python implementation of an incremental Structure from Motion (SfM) pipeline. It reconstructs a 3D scene and camera poses from a sequence of 2D images having varying camera intrinsics. The entire pipeline is built from scratch using **OpenCV**, **NumPy**, and **SciPy**, with **Open3D** for visualization.

The pipeline also has implementation of  global **Bundle Adjustment (BA)** procedure that jointly optimizes 3D point clouds and camera parameters for improved accuracy.

<!-- ![Final Reconstruction GIF](https://raw.githubusercontent.com/AhaanBanerjee/Visual-Odometry-and-SLAM-Implementations/master/artefacts/SfM.gif)
*(Example output from the Temple dataset)* -->

## The SfM Workflow

The pipeline follows these sequential steps to build the 3D model:

1.  **Initialization**:
    *   An optimal "seed pair" of two initial images is selected based on a good trade-off between baseline and feature overlap.
    *   Features are matched between these two images.
    *   The relative camera pose is estimated, and initial 3D points are triangulated.

2.  **Incremental Processing (Main Loop)**:
    *   **Image Selection**: The next best image to add to the reconstruction is selected. The "best" image is the one that observes the most existing 3D points.
    *   **Camera Pose Estimation (PnP)**: The pose of the new camera is estimated by finding 2D-3D correspondences between its features and the existing 3D map. PnP with RANSAC is used for the estimation
    *   **New Point Triangulation**: New 3D points are created by triangulating features between the newly added view and other existing views.
    *   **Data Filtering**: At every stage, filtering is applied to remove poor-quality 3D points based on reprojection error and triangulation angle (parallax). This is crucial for stability.

3.  **Optimization (Bundle Adjustment)**:
    *   **Incremental BA**: After every *N* new images are added (e.g., N=5), a global bundle adjustment is run to minimize reprojection error across all views and points seen so far. This corrects for accumulated drift.
    *   **Final BA**: After all images have been processed, a final, more intensive bundle adjustment is performed with more iterations to produce the final, highly accurate reconstruction.

4.  **Visualization**:
    *   The 3D point cloud and estimated camera poses are visualized in real-time throughout the process, providing immediate feedback on the reconstruction quality.

## Datasets used

<!-- ### ETHZ 3D - Facade Dataset
<!-- Placeholder: Insert an image of your initial 2-view reconstruction here -->
<!-- ![Initial Reconstruction](/results/Facade_results/ScreenCapture_2025-08-28-02-51-32.png) --> -->


### Fountain Dataset
Total 3D points- 16908, Mean Reprojection Error - 0.409
<!-- Placeholder: Insert an image of your mid-reconstruction point cloud and cameras here -->
![Intermediate Reconstruction GIF](/src/artefacts/fountain2_up.gif)

### Temple Ring Dataset
Total 3D points- 20467, Mean Reprojection Error - 0.356
<!-- Placeholder: Insert an image of your mid-reconstruction point cloud and cameras here -->
![Intermediate Reconstruction GIF](/src/artefacts/temple.mp4.gif)


## How to Run

1.  **Clone the repository:**
    ```
    git clone https://github.com/AhaanB29/Structure_from_Motion/tree/main
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```
    pip install numpy opencv-python scipy open3d matplotlib pandas
    ```

3.  **Configure the `Reconstruction.py` script:**
    *   Inside the `if __name__ == '__main__':` block, update the `img_path` variable to point to your image dataset.
    *   Set the camera intrinsics matrix `K` (if working with single camera)
    *   Enter the path for the camera parameters as csv file (if working with multi-camera) 
    *   Use the seed_pair function to get the seed pair

4.  **Run the pipeline:**
    ```
    python Reconstruction.py
    ```
    An Open3D window will pop up to visualize the reconstruction process and the 3D plots will be saved as '.ply'