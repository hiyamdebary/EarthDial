# 📌 Evaluation Pipeline for Image Classification

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Image Classification** task.

---

## 📂 Data Preparation

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p src/earthdial/eval/data/rs_classification
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
 ├── AID
 ├── LCZs_S2
 ├── TreeSatAI
 ├── UCM
 ├── WHU_19 
 ├── BigEarthNet_FINAL_RGB
 ├── BigEarthNet_S2
 ├── STARCOP_test
 ├── UHI_test
```

---

## 🚀 Running the Evaluation

To execute the evaluation process on an **8-GPU setup**, run the following command:

```shell
# Test the RGB-Datasets: 'AID,UCM,WHU_19,BigEarthNet_FINAL_RGB'
GPUS=8 ./src/earthdial/eval/eval.sh ./checkpoints/EarthDial_4B_RGB rs_classification --dynamic

# Test the MS-Datasets: 'LCZs_S2,TreeSatAI,BigEarthNet_S2'
GPUS=8 ./src/earthdial/eval/eval.sh ./checkpoints/EarthDial_4B_MS rs_classification --dynamic

# Test the MS-Datasets: 'STARCOP_test,UHI_test'
GPUS=8 ./src/earthdial/eval/eval.sh ./checkpoints/EarthDial_4B_Methane_UHI rs_classification --dynamic
```

This tests our EarthDial-4B for classification task, saves result files (e.g., `src/earthdial/eval/rs_classification/results/AID.jsonl`) and displays the classification accuracy.

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

