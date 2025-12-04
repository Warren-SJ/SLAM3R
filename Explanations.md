# SLAM3R Model Explanations

This document explains the architecture, components, and configuration of the SLAM3R model as implemented in this project.

## Core Architecture

The system utilizes two primary deep learning models, both inheriting from a common `Multiview3D` backbone (based on DUSt3R/MASt3R architectures).

### 1. Image2PointsModel (I2P)

* **Role:** **Local Reconstruction**. This model is responsible for taking a small window of adjacent video frames and reconstructing the 3D geometry (point clouds) in a **local coordinate system** (usually centered on a reference frame within the window).
* **Mechanism:**
  * It uses a Transformer-based **Encoder** to extract features ("image tokens") from RGB images.
  * It uses a **Decoder** that employs cross-attention to exchange information between the "reference view" and "source views".
  * It predicts **3D pointmaps** and **confidence maps** directly.
  * It includes a **Retrieval Module** (`get_corr_score`) to calculate correlation scores between views, which helps in selecting the best frames (keyframes) to process.
* **Input:** A set of image frames (window).
* **Output:** 3D point clouds for these frames, aligned relative to one of the input frames.

### 2. Local2WorldModel (L2W)

* **Role:** **Global Registration**. This model aligns locally reconstructed frames (from I2P) into a consistent **global "World" coordinate system**.
* **Mechanism:**
  * It takes two types of inputs:
    1. **Reference Views (Scene Frames):** Frames that are already registered in the global system (with known global 3D points).
    2. **Source Views (Keyframes):** New frames that have local 3D points (from I2P) but need to be registered to the global system.
  * It uses a **Point Embedder** (Conv2d) to embed the input 3D pointmaps before passing them to the decoder.
  * The model refines the reference points and predicts the transformation/alignment for the source points to place them into the global coordinate space.
* **Input:** Global scene frames + new local keyframes.
* **Output:** All frames registered in the global coordinate system with updated confidence scores.

### 3. Attention Mechanism Details

The models utilize a specialized **Multiview Decoder Block** (`MultiviewDecoderBlock_max`) to handle interactions between frames. While the block architecture is shared, the interaction topology differs between the two models.

#### General Mechanism

* **Self-Attention:** Applied individually to each view (both reference and source) to refine features based on internal spatial context, using **RoPE** for 2D positional encoding.
* **Aggregation (Max-Pooling):** When a view attends to multiple other views, the retrieved features are aggregated using **Max-Pooling**. This allows the model to selectively retain the most relevant features from the pool of available views.

#### Model-Specific Attention Topologies

* **I2P Attention (Local Reconstruction)**
  * **Structure:** Star-like topology centered on the Reference frame.
  * **Ref $\to$ Srcs (One-to-Many):** The single local reference frame queries features from *all* neighboring source frames in the window and aggregates them (via Max-Pooling). This gathers context to build the central coordinate system.
  * **Src $\to$ Ref (Many-to-One):** Each source frame queries features *only* from the central reference frame. This ensures all source frames align themselves to the reference.

* **L2W Attention (Global Registration)**
  * **Structure:** Bipartite graph (Many-to-Many) between Global Scene Frames and New Keyframes.
  * **Geometric Guidance:** Unlike I2P, the L2W model injects **Point Embeddings (PE)** into the decoder. These embeddings are derived from the input 3D point clouds (global points for Ref, local points for Src), providing geometric cues to guide the attention.
  * **Ref $\to$ Srcs (Many-to-Many):** Each global scene frame queries all new keyframes to update its representation.
  * **Src $\to$ Refs (Many-to-Many):** Each new keyframe queries *all* global scene frames in the current batch. This is crucial for registering the new frame against the existing global map.

## Reconstruction Pipeline

The reconstruction process in `recon.py` follows these steps:

1. **Preprocessing:** Extracts image tokens for all frames using the shared Encoder (efficient, done once).
2. **Initialization:**
    * Selects an initial window of frames (`--initial_winsize`).
    * Uses **I2P** to reconstruct this initial segment and establish the "World" coordinate system (Frame 0 usually defines the origin).
    * These initial frames form the start of the **Buffering Set** (Global Map).
3. **Local Reconstruction Loop (I2P):**
    * Iterates through the video with a specific stride (`--keyframe_stride`).
    * For each keyframe, it selects a local window of neighbors (radius `--win_r`).
    * Runs **I2P** to get 3D points in that keyframe's local space.
4. **Global Registration Loop (L2W):**
    * The system maintains a **Buffering Set** of high-quality "scene frames" that represent the global map.
    * It selects the best matching scene frames from this buffer (using retrieval scores).
    * Runs **L2W** to register the new locally reconstructed keyframes against these selected global scene frames.
    * Buffer Update:** New registered frames are added to the buffer. Old frames might be removed based on the strategy (`reservoir` or `fifo`) to keep the buffer size manageable (`--buffer_size`).

## Tuning Parameters

These parameters in `recon.py` allow you to trade off between speed, accuracy, and memory usage.

### Keyframe & Windowing Strategy

* **`--keyframe_stride`** (Default: -1 [Auto])
  * **Description:** The step size for selecting main keyframes to reconstruct.
  * **Tuning:**
    * **-1 (Auto):** The system runs a test (`adapt_keyframe_stride`) to find the stride that gives the best confidence/overlap.
    * **Small (e.g., 2-5):** High overlap, very dense reconstruction, slower processing.
    * **Large (e.g., 10+):** Faster, covers large areas quickly, but risk of tracking loss or lower density.
* **`--win_r`** (Default: 3)
  * **Description:** The radius of the local window for I2P. A radius of 3 means the window size is $3 \times 2 + 1 = 7$ frames.
  * **Tuning:** Larger radius = more multiview consistency but higher VRAM usage.

### Confidence Thresholds

* **`--conf_thres_i2p`** (Default: 1.5)
  * **Description:** Threshold for the local I2P model.
  * **Effect:** Points with confidence below this are discarded before being passed to L2W.
  * **Tuning:** Increase this to remove noisy outliers early on; decrease if the scene is sparse or texture-less.
* **`--conf_thres_l2w`** (Default: 12)
  * **Description:** Threshold for the final output.
  * **Effect:** Only points with global confidence higher than this are saved to the final `.ply` file.
  * **Tuning:** Increase for a cleaner, higher-precision point cloud. Decrease to fill holes in difficult areas.

### Global Buffer & Memory

* **`--buffer_size`** (Default: 100)
  * **Description:** Maximum number of frames kept in the "Global Map" buffer for registration.
  * **Tuning:**
    * **Larger:** Better global consistency and loop closure (if supported), but higher memory/VRAM usage.
    * **Smaller:** Lower memory footprint, acts more like visual odometry (local tracking).
* **`--buffer_strategy`** (Default: 'reservoir')
  * **Description:** How to replace frames when buffer is full.
  * **Options:**
    * `reservoir`: Randomly replaces frames, mathematically ensuring a representative sample of the *entire* history. (Best for long videos).
    * `fifo`: First-In-First-Out. Keeps only the most recent frames. (Good for simple forward motion, bad for revisiting areas).
* **`--num_scene_frame`** (Default: 10)
  * **Description:** How many reference frames are actually retrieved from the buffer to register *one* batch of new frames.
  * **Tuning:** Higher = more robust registration, slower inference.

### Output Quality

* **`--num_points_save`** (Default: 2,000,000)
  * **Description:** The target number of points in the final `_recon.ply` file.
  * **Effect:** The system performs reservoir sampling on the valid points to hit this target count exactly.
