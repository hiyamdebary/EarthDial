# 📌 Evaluation Pipeline for Image Captioning

## 🌟 Overview
This repository provides a streamlined **evaluation pipeline** for the **Image Captioning** task.

---

## 📂 Data Preparation

Before downloading the datasets, ensure that the following directory structure exists:

```shell
mkdir -p ./eval/data/rs_caption
```

### 📸 Image Captioning Datasets
Follow these steps to prepare the datasets:

```shell
# Step 1: Navigate to the data directory
cd ./eval/data/rs_caption

# Step 2: Download dataset shard files

cd ../..
```

After preparation, the expected directory structure will be:

```shell
./eval/data/rs_caption
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
# Test the rs_caption datasets
GPUS=8 sh eval.sh checkpoints/EarthDial_4B image_captioning --dynamic
```

After testing, a results file (e.g., `results/NWPU_RESISC45_Captions.jsonl`) will be generated.

To evaluate the **rs_caption datasets**, run:

```shell
python caption_eval.py
```

---

## 📌 Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script parameters if needed to match your system's configuration.
- Contributions & improvements are welcome! 🚀

