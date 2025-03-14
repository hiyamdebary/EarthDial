# 📌 Evaluation Pipeline for Image Captioning

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Image Captioning** task.

---

## 📂 Data Preparation

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p ./validation_data/Image_captioning/
```

### 📸 Image Captioning Datasets
Follow these steps to prepare the datasets:

```shell
# Step 1: Navigate to the data directory
cd ./validation_data/Image_captioning/

# Step 2: Download dataset shard files

cd ../..
```

After preparation, the expected directory structure will be:

```shell
./validation_data/Image_captioning/
 ├── NWPU_RESISC45_Captions
 ├── RSICD_Captions
 ├── RSITMD_Captions
 ├── sydney_Captions
 ├── UCM_Captions
```

---

## 🚀 Running the Evaluation

To execute the evaluation process on an **8-GPU setup**, run the following command:

```shell
# Test the rs_image_caption datasets
GPUS=8 ./src/earthdial/eval/eval.sh rs_image_caption --dynamic
```

This tests our EarthDial-4B on for image captioning task, saves result files (e.g., `src/earthdial/eval/rs_image_caption/results/NWPU_RESISC45_Captions.jsonl`) and displays the captioning score.

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

