# 📌 Evaluation Pipeline for Region Captioning

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Region Captioning** task.

---

## 📂 Data Preparation

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p ./validation_data/Region_captioning/
```

### 📸 Image Captioning Datasets
Follow these steps to prepare the datasets:

```shell
# Step 1: Navigate to the data directory
cd ./validation_data/Region_captioning/

# Step 2: Download dataset shard files

cd ../..
```

After preparation, the expected directory structure will be:

```shell
./validation_data/Region_captioning/
 ├── GeoChat
 ├── HIT_UAV_test
 ├── NWPU_VHR_10_test
 ├── ship_dataset_v0_test
 ├── SRSDD_V1_0_test
 ├── Swimming_pool_dataset_test
 ├── UCAS_AOD
 ├── urban_tree_crown_detection
```

---

## 🚀 Running the Evaluation

To execute the evaluation process on an **8-GPU setup**, run the following command:

```shell
# Test the rs_image_caption datasets
GPUS=8 ./src/earthdial/eval/eval.sh rs_region_captioning --dynamic
```

This tests our EarthDial-4B on for region captioning task, saves result files (e.g., `src/earthdial/eval/rs_region_captioning/results/GeoChat.jsonl`) and displays the region captioning score.

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

