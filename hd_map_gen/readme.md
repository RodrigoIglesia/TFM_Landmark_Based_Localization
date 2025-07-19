# HD Map Generator

This module provides tools for generating high-definition (HD) maps for autonomous vehicle localization using Waymo Open D# 🗺️ Waymo HD Map Generator

This module parses scenes from the **Waymo Open Dataset**, combining 3D point clouds and semantic segmentation labels to enrich HD maps with **traffic sign information**. It integrates **LiDAR processing**, **pose transformation**, **clustering**, and **protobuf editing** to create updated map representations.

---

## ⚙️ Key Features

- Extracts and transforms labeled 3D point clouds from Waymo dataset frames.
- Filters LiDAR points to detect traffic signs via segmentation labels.
- Accumulates vehicle poses to express all point clouds in a global frame.
- Clusters traffic sign points and computes centroids.
- Enriches the original **FeatureMap** with detected signs.
- Exports results to:
  - `.json` format (protobuf → JSON)
  - `.csv` files (centroid positions)
  - *(optional)* `.tfrecord` with updated frames (commented out)

---

## 🧭 Workflow Summary

For each Waymo scene:

1. **Extract Feature Map** (from the first frame).
2. **Parse LiDAR Point Clouds** with segmentation labels.
3. **Filter Sign-Labeled Points**, project them into global coordinates.
4. **Cluster Points** to identify individual traffic signs.
5. **Compute Centroids** and enrich the Feature Map with stop sign data.
6. **Export Results**:
   - JSON version of the new map
   - CSV file with sign coordinates
   - *(Optional)* Updated TFRecord file with modified map features.

---

## 🧪 Input Requirements

- A Waymo TFRecord scene with:
  - Map features in the first frame.
  - 3D LiDAR segmentation labels.
- Waymo dataset parsing dependencies:
  - `waymo_open_dataset`
  - `tensorflow`
  - `protobuf`
- Utility modules in `waymo_utils/`:
  - `WaymoParser`, `waymo_3d_parser`, `transform_utils`

---

## 🗂️ File Outputs

- `signs_map_features_<scene>.json`: Enriched feature map with traffic signs.
- `signs_map_features_<scene>.csv`: Centroids (X, Y, Z) of signs.
- `pointcloud_concatenated<scene>.csv`: All concatenated filtered point clouds.
- *(Optional)* `output<scene>.tfrecord`: TFRecord with updated map features.

---

## 🚀 Running the Script

```bash
python map_pointcloud_concatenated.py
ataset data.

## Objective

The goal of this module is to automate the creation of accurate HD maps, which are essential for precise vehicle localization and navigation in autonomous driving applications.

## Usage

1. Clone the repository and navigate to the module directory:
    ```bash
    git clone <repo_url>
    cd WaymoLoc/hd_map_gen
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the map generation script:
    ```bash
    python generate_hd_map.py --input <waymo_data_path> --output <map_output_path>
    ```

Refer to the script's help (`-h`) for additional options.
