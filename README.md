# SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos

This repository contains a study of the paper [SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_SLAM3R_Real-Time_Dense_Scene_Reconstruction_from_Monocular_RGB_Videos_CVPR_2025_paper.pdf) by Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnam Fan and  Baoquan Chen. This paper is a CVPR 2025 highlight paper.

## Introduction

SLAM3R is a novel real-time end-to-end dense 3D reconstruction system that uses RGB videos to directly predict 3D pointmaps in a unified coordinate system through feed-forward neural networks.

Previous techniques typically require offline processing which limits their ability to be used in real-time applications. Dense scene reconstruction systems have been developed. However, these all short in terms of accuracy or completeness or they require depth sensors. monocular SLAM (Simultaneous Localization and Mapping) systems have been proposed, however these come at the cost of reduced runtime efficiency. The goal of this research paper is to develop a model which satisfies the three criteria: **reconstruction accuracy**, **completeness** and **runtime efficiency**.

The network consists of two main components:

1. **Image-to-points (I2P)**: Reconstructs local geometry from a sliding window

2. **Local-to-World (L2W)**: Registers local reconstructions to build a globally consistent 3D scene

## Related Work

* Traditional offline approaches: Classical approaches first determine camera parameters using structure from motion (SfM) followed by dense 3D point triangulation using multi-view stereo (MVS). Neural implicit and 3D Gaussian representations have been applied to enhance the quality of dense reconstruction. Due to offline processing, these methods are not suitable for real-time applications.

* Dense SLAM: Early approaches prioritized real-time performance but produced only sparse structures of the scene. Dense SLAM approaches incorporate detailed scene geometry information to improve pose estimation. More recent approaches monocular dense SLAM systems have been proposed. All methods alternate between solving for camera poses and estimating the scene representation.

* End-to-end dense 3D reconstruction: Recent works have explored end-to-end learning-based approaches for dense 3D reconstruction. These methods typically use deep neural networks to directly predict 3D structures from input images. Dust3R is purely end-to-end without relying on camera parameters. MAST3R adds a match head. Spann3R adds spatial memory by performing incremental scene reconstruction in a unified coordinate system.
