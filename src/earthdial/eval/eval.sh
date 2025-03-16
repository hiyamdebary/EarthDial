#!/bin/bash

set -x

CHECKPOINT=${1}
TASK=${2}
export PYTHONPATH="${PYTHONPATH}"
#echo "CHECKPOINT: ${CHECKPOINT}"
echo "CHECKPOINT: ${TASK}"

MASTER_PORT=${MASTER_PORT:-6391}
PORT=${PORT:-63709}
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
export PATH=$(echo $PATH | sed -e 's|:/home/muzammal/.local/bin||g')
export PYTHONPATH=./src/earthdial/eval/:$PYTHONPATH


ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"



if [ "${TASK}" = "rs_classification" ]; then
    
    DATASETS='STARCOP_test,UHI_test'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_classification/classification_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_classification/results "${ARGS[@]:2}"

    python src/earthdial/eval/rs_classification/eval.py --datasets ${DATASETS}
fi



if [ "${TASK}" = "rs_classification_MS" ]; then
    
    DATASETS='rs_LCZ_test,TreeSatAI,BigEarthNet_S2'
    CHECKPOINT='./checkpoints/EarthDial_4B_MS'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_classification/classification_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_classification/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_classification/eval.py --datasets ${DATASETS}
fi



if [ "${TASK}" = "rs_image_caption" ]; then
    
    DATASETS='NWPU_RESISC45_Captions,RSICD_Captions,RSITMD_Captions,sydney_Captions,UCM_captions'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_image_caption/captioning_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_image_caption/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_image_caption/eval.py --datasets ${DATASETS}
fi



if [ "${TASK}" = "rs_detection_RGB" ]; then
    
    DATASETS='GeoChat,NWPU_VHR_10,Swimming_pool_dataset,urban_tree_crown_detection'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_detection/detection_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_detection/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_detection/eval.py --datasets ${DATASETS}
fi

if [ "${TASK}" = "rs_detection_MS" ]; then
    
    DATASETS='ship_dataset_v0'
    CHECKPOINT='./checkpoints/EarthDial_8B_MS'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_detection/detection_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_detection/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_detection/eval.py --datasets ${DATASETS}
fi



if [ "${TASK}" = "rs_region_captioning" ]; then
    
    DATASETS='GeoChat,HIT_UAV_test,NWPU_VHR_10_test,ship_dataset_v0_test,SRSDD_V1_0_test,Swimming_pool_dataset_test,UCAS_AOD,urban_tree_crown_detection'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_region_captioning/captioning_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_region_captioning/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_region_captioning/eval.py --datasets ${DATASETS}
fi


if [ "${TASK}" = "rs_grounding_description" ]; then
    
    DATASETS='HIT_UAV,NWPU_VHR_10,Swimming_pool_dataset,UCAS_AOD'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_grounding_description/grounding_desscription_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_grounding_description/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_grounding_description/eval.py --datasets ${DATASETS}
fi


if [ "${TASK}" = "rs_vqa" ]; then
    
    DATASETS='RSVQA_LR,RSVQA_HR'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_vqa/vqa_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_vqa/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_vqa/eval.py --datasets ${DATASETS}
fi


if [ "${TASK}" = "rs_change_detection" ]; then
    

    #Change detection datasets: Dubai_CC,LEVIR_MCI,MUDS,SYSU

    #Image Captioning dataset: xBD_image_captioning
    #Region Classification dataset: xBD_reg_cls_testset_1,xBD_reg_cls_testset_2
    #Image Classification Task Datasets: FMoW,xBD_testset_1,xBD_testset_2,xBD_testset_3
    #Object Detection dataset: xBD_object_detection
    #Reffered Object Detection dataset: xBD_referred_object_detection
    
    DATASETS='Dubai_CC,LEVIR_MCI,MUDS,SYSU'
    CHECKPOINT='./checkpoints/EarthDial_4B_RGB'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_change_detection/rs_change_detection_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_change_detection/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_change_detection/eval_detection.py --datasets ${DATASETS}
    #python src/earthdial/eval/rs_change_detection/eval_detection.py --datasets ${DATASETS}
    python src/earthdial/eval/rs_change_detection/eval_classification.py --datasets ${DATASETS}
    python src/earthdial/eval/rs_change_detection/eval_caption.py --datasets ${DATASETS}
fi


if [ "${TASK}" = "rs_methane_plume" ]; then

    DATASETS='rs_UHI'
    CHECKPOINT='./checkpoints/EarthDial_4B_Methane_UHI'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/EarthDial/src/earthdial/eval/rs_methane_plume/classification_shards_MS_UHI.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_methane_plume/results "${ARGS[@]:2}"
fi





























if [ "${DATASET}" = "MS_detection_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/MS_detection_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_detection/MS_shards_results/4B_Full_8Nov_Pretrained_MS_MLP_LLM "${ARGS[@]:2}"
fi





if [ "${DATASET}" = "quakeset_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/classification_shards_Quakeset.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/quake_set/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change_MS_Quakset_2 "${ARGS[@]:2}"
fi


























if [ "${DATASET}" = "UHI" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/classification_shards_MS_UHI.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_UHI/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change_MS_UHI_Methane "${ARGS[@]:2}"
fi

