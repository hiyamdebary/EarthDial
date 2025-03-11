#!/bin/bash

set -x

CHECKPOINT=${1}
TASK=${2}
export PYTHONPATH="${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"
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
    
    DATASETS='AID,UCM,WHU_19,BigEarthNet_RGB,rs_LCZ_test,TreeSatAI,BigEarthNet_S2'

    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    src/earthdial/eval/rs_classification/classification_test.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --out-dir src/earthdial/eval/rs_classification/results "${ARGS[@]:2}"

    clear && python src/earthdial/eval/rs_classification/eval.py --datasets ${DATASETS}
fi


if [ "${DATASET}" = "detection_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_detection/detection_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_detection/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "identify_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_identify/identify_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_identify/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "grounding_description" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_description/rs_grounding_desscription.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_description/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "caption_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_caption/captioning_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_caption/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "vqa-rs-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_vqa/rs_vqa.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_vqa/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "change_detection" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_change_detection/change_detection_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_change_detection/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
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

if [ "${DATASET}" = "MS_classification_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_classification/classification_shards_MS_LCZ.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_classification/MS_shards_results/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change_MS "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "methane_plume" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/classification_shards_Methan.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_methane/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change_MS_UHI_Methane "${ARGS[@]:2}"
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




if [ "${DATASET}" = "GeoBench_classification_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval_copy/rs_classification/GeoBench_classification_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval_copy/rs_GeoBench_classification/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "GeoBench_change_detection" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval_copy/GEOBench_change_detection/change_detection_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval_copy/GEOBench_change_detection/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
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





















if [ "${DATASET}" = "ms_grounding_description" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/MS_grounding_description_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_description/8B_Full_3Nov_RGBAll_MLP_LLM_1 "${ARGS[@]:2}"
fi

























if [ "${DATASET}" = "classification" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_classification/classification.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_classification/results_speed "${ARGS[@]:2}"
fi





if [ "${DATASET}" = "GEOBench_detection_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/GEOBench_detection/detection_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/GEOBench_detection/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi







if [ "${DATASET}" = "MS_identify_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/MS_identify_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_identify/ms_shards_results/4B_Full_8Nov_Pretrained_MS_MLP_LLM "${ARGS[@]:2}"
fi











if [ "${DATASET}" = "Geobench_caption_shards" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/GEOBench_captioning/captioning_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/GEOBench_captioning/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change "${ARGS[@]:2}"
fi






if [ "${DATASET}" = "ms_grounding_description" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/GeoVLM_Git/src/geovlm/eval/rs_classification/MS_grounding_description_shards.py --checkpoint ${CHECKPOINT} --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_description/8B_Full_3Nov_RGBAll_MLP_LLM_1 "${ARGS[@]:2}"
fi








if [ "${DATASET}" = "ucmerced-rs-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /l/users/fahad.khan/akshay/mbzuai_ibm/InternVL/internvl_chat/eval/rs_classification/classification.py --checkpoint ${CHECKPOINT} --datasets rs_ucmerced_test --out-dir /l/users/fahad.khan/akshay/mbzuai_ibm/InternVL/internvl_chat/eval/rs_classification/results "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "identify-rs-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_grounding/rs_evaluate_identify.py --checkpoint ${CHECKPOINT} --datasets ref_geochat_val --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_grounding/results_identify "${ARGS[@]:2}"
fi

if [ "${DATASET}" = "rs_ref_geochat-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_grounding/rs_evaluate_refer.py --checkpoint ${CHECKPOINT} --datasets ref_geochat_val --out-dir /share/data/drive_2/remote_sensing/InternVL/internvl_chat/eval/rs_grounding/results_refer "${ARGS[@]:2}"
fi