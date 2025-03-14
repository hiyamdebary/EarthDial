# 📌 Evaluation Pipeline for Object Detection

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Object Detection** task.

---

## 📂 Data Preparation

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p src/earthdial/eval/data/rs_detection
```

### 📸 Image Classification Datasets
Follow these steps to prepare the datasets:

```shell
# Step 1: Navigate to the data directory
cd EarthDial/validation_data

# Step 2: Download dataset shard files


```

After preparation, the expected directory structure will be:

```shell
EarthDial/validation_data/
 ├── GeoChat
 ├── NWPU_VHR_10
 ├── Swimming_pool_dataset
 ├── urban_tree_crown_detection
```

---

## 🚀 Running the Evaluation

To execute the evaluation process on an **8-GPU setup**, run the following command:

```shell
# Test the rs_classification datasets
GPUS=8 ./src/earthdial/eval/eval.sh rs_detection --dynamic
```

This tests our EarthDial-4B on for object detection task, saves result files (e.g., `src/earthdial/eval/rs_detection/results/GeoChat.jsonl`) and displays the object detection scores.

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

