# --------------------------------------------------------
# GEOVLM
# Copyright (c) 2024 
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import os
import json
import logging
import math
import tempfile
import sys
import warnings
import numpy as np
import torch
import torch.distributed as dist
import transformers
from datasets import load_from_disk, concatenate_datasets
import gc
# Set the environment variable
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from copy import deepcopy

# Adjust path based on script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Base path to add: {base_dir}")
path_to_other_project = os.path.join(base_dir, "../../")
absolute_path = os.path.abspath(path_to_other_project)
if not os.path.exists(absolute_path):
    print(f"Path does not exist: {absolute_path}")
# Add the directory to the system path
sys.path.append(absolute_path)

# import packages
from earthdial.dist_utils import init_dist
from earthdial.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from earthdial.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
from earthdial.patch import (
    concat_pad_data_collator,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_train_sampler,
)
from earthdial.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    S2_RGB_10_TOKEN,
    L8_RGB_30_TOKEN,
    HIGH_RGB_05_TOKEN,
    HIGH_RGB_05_TEMP_TOKEN,
    S2_MS_10_TOKEN,
    HIGH_RGBI_05,
    S1_VH_10_TOKEN,
    S1_VH_1_TOKEN,
    TREECLASSIFY,
    GROUNDING,
    REFER,
    CLASSIFY,
    IDENTIFY,
    CAPTION,
    CHANGEDET,
)
from earthdial.train.dataset import (
    ConcatDataset,
    TCSLoader,
    WeightedConcatDataset,
    build_transform,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3,
)
from earthdial.train.trainer_monkey_patch import replace_create_optimizer
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
#from dataloader import ShardDataLoader
# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

has_tcs_loader = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from huggingface_hub import snapshot_download

#from dataloader_pretrain import ShardDataLoader

from dataloader_pretrain import ShardDataLoader_pretrain
import json
import math
import concurrent.futures
from torch.utils.data import ConcatDataset
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM decoder."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the MLP layers of the model."},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={
            "help": "Specify the number of ViT layers to unfreeze. Default is 0."
        },
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={
            "help": "Specify the layer of ViT feature map to use. Default is last layer."
        },
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={
            "help": "Set the LoRA adapter rank for the backbone model. Default is 0."
        },
    )
    use_llm_lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={"help": "Set to True to enable the use of a custom trainer."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Set the drop path rate for the ViT model. Default is 0."},
    )
    ps_version: str = field(
        default="v2",
        metadata={
            "help": "Specify the version of pixel shuffle implementation. Default is `v1`."
            "Please use `v2` to fix the bug of transposed image."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Set the desired down-sampling ratio for the image. Default is 1.0."
        },
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={"help": "Pad the image to a square shape if set to True."},
    )
    conv_style: Optional[str] = field(
        default="internlm2-chat", metadata={"help": "Prompt style for a conversation."}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the meta file of datasets."},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use data resampling."},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use dynamic image size."},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to add a thumbnail image."},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum number of dynamic patches. Default is 1."},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={"help": "The maximum number of dynamic patches. Default is 6."},
    )
    normalize_type: Optional[str] = field(
        default="imagenet",
        metadata={"help": "The normalize type for the image. Default is imagenet."},
    )

def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type="imagenet"
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
   # logger.info(f"Reading JSON {data_args.meta_path} file")
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
       # logger.info(f"Reading JSON ID {ds_idx} file")
       # logger.info(f"Reading JSON ds_name {ds_name} file")
        repeat_time = ds_collections[ds_name]["repeat_time"]
        if "max_dynamic_patch" in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]["max_dynamic_patch"]
            logger.info(
                f"max_dynamic_patch is set to {max_num} according to the meta file"
            )
        else:
            max_num = max_dynamic_patch
        if "dynamic_image" in ds_collections[ds_name]:
            dynamic_image_size = ds_collections[ds_name]["dynamic_image"]
            logger.info(
                f"dynamic_image is set to {dynamic_image_size} according to the meta file"
            )
        else:
            max_num = max_dynamic_patch
        dataset = ShardDataLoader_pretrain(
            model,
            logger,
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]["data_augment"],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
            )
        datasets.append(dataset)
        logger.info(f"Added dataset: {ds_name} with length: {len(dataset)}")
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset

# def load_dataset(
#     logger, ds_idx, ds_name, ds_collections, tokenizer, tcs_loader, model, data_args,
#     group_by_length, dynamic_image_size, use_thumbnail, min_dynamic_patch, max_dynamic_patch, normalize_type
# ):
#     repeat_time = ds_collections[ds_name]["repeat_time"]
#     if "max_dynamic_patch" in ds_collections[ds_name]:
#         max_num = ds_collections[ds_name]["max_dynamic_patch"]
#         logger.info(
#             f"max_dynamic_patch is set to {max_num} according to the meta file"
#         )
#     else:
#         max_num = max_dynamic_patch
#     if "dynamic_image" in ds_collections[ds_name]:
#         dynamic_image_size = ds_collections[ds_name]["dynamic_image"]
#         logger.info(
#             f"dynamic_image is set to {dynamic_image_size} according to the meta file"
#         )
#     else:
#         dynamic_image_size = dynamic_image_size

#     dataset = ShardDataLoader_pretrain(
#         logger,
#         data_args.conv_style,
#         ds_collections[ds_name],
#         tokenizer,
#         tcs_loader,
#         ds_name=ds_name,
#         num_image_token=model.num_image_token,
#         image_size=data_args.force_image_size,
#         is_train=ds_collections[ds_name]["data_augment"],
#         pad2square=data_args.pad2square,
#         group_by_length=group_by_length,
#         dynamic_image_size=dynamic_image_size,
#         use_thumbnail=use_thumbnail,
#         min_dynamic_patch=min_dynamic_patch,
#         max_dynamic_patch=max_num,
#         repeat_time=repeat_time,
#         normalize_type=normalize_type,
#         random_seed=ds_idx,
#     )
#     logger.info(f"Added dataset: {ds_name} with length: {len(dataset)}")
    
#     if data_args.use_data_resampling:
#         length = math.sqrt(len(dataset))
#     else:
#         length = len(dataset)
    
#     return dataset, length

# def build_datasets(
#     data_args,
#     tokenizer,
#     tcs_loader,
#     model,
#     group_by_length=False,
#     dynamic_image_size=False,
#     use_thumbnail=False,
#     min_dynamic_patch=1,
#     max_dynamic_patch=12,
#     normalize_type="imagenet"
# ):
#     # Load the JSON meta file
#     ds_collections = json.loads(open(data_args.meta_path).read())
    
#     datasets = []
#     lengths = []

#     # Use ThreadPoolExecutor to load datasets in parallel
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         futures = []
        
#         # Submit each dataset loading task to the thread pool
#         for ds_idx, ds_name in enumerate(ds_collections.keys()):
#             futures.append(
#                 executor.submit(
#                     load_dataset,
#                     logger, ds_idx, ds_name, ds_collections, tokenizer, tcs_loader, model, 
#                     data_args, group_by_length, dynamic_image_size, use_thumbnail, 
#                     min_dynamic_patch, max_dynamic_patch, normalize_type
#                 )
#             )
        
#         # Collect the results as they complete
#         for future in concurrent.futures.as_completed(futures):
#             dataset, length = future.result()
#             datasets.append(dataset)
#             lengths.append(length)

#     # Combine datasets based on whether data resampling is used
#     if data_args.use_data_resampling:
#         total_length = sum(lengths)
#         weights = [l / total_length for l in lengths]
#         train_dataset = WeightedConcatDataset(datasets, weights)
#     else:
#         train_dataset = ConcatDataset(datasets)

#     return train_dataset

def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get("LAUNCHER", "pytorch")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 
    
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # # Define the custom log filter
    # class SuppressMicrostepLogs(logging.Filter):
    #     def filter(self, record):
    #         # Filter out messages containing 'fwd_microstep'
    #         return 'fwd_microstep' not in record.getMessage()

    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # # Add the custom filter to the root logger
    # logger = logging.getLogger()
    # logger.addFilter(SuppressMicrostepLogs())

    # # Apply the filter to specific loggers like 'transformers' or 'deepspeed'
    # transformers_logger = logging.getLogger("transformers")
    # transformers_logger.addFilter(SuppressMicrostepLogs())

    # deepspeed_logger = logging.getLogger("deepspeed")
    # deepspeed_logger.addFilter(SuppressMicrostepLogs())

    # # Ensure that the logging level is set based on the training arguments
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)  # Apply the log level from training args
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # # if training_args.should_log is True, ensure that log level is info
    # if training_args.should_log:
    #     transformers.utils.logging.set_verbosity_info()

    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()
    # #training_args.logging_steps = 500
    # log_level = training_args.get_process_log_level()

    # #print("Log level",log_level)
    # logger.setLevel(log_level)
    # #logger.setLevel(logging.WARNING)  # Set log level to WARNING
    # set_verbosity(log_level)
    # enable_default_handler()
    # enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path

    #download model
   # Define the model repository and the destination directory
  #  destination_dir = "/cos/Model_Files/Model_Weights/pretrained_models/8B_model"
    # destination_dir= "/cos/Model_Weights/4B_model/"
    # # Download the model snapshot
    # snapshot_download(
    #         repo_id=model_args.model_name_or_path,                     # Model repository
    #         local_dir=destination_dir,              # Target directory
    #         resume_download=True,                   # Resume interrupted downloads
    #         local_dir_use_symlinks=False            # Do not use symbolic links
    #     )

    # print(f"Model downloaded to this location: {destination_dir}")
    # temp_dir = "/cos/Model_Files/Model_Weights/cache/"
    # #tempfile.gettempdir()
    # print(f"Clearing temporary files from: {temp_dir}")
    # if os.path.exists(temp_dir):
    #     for item in os.listdir(temp_dir):
    #         item_path = os.path.join(temp_dir, item)
    #         try:
    #             if os.path.isfile(item_path) or os.path.islink(item_path):
    #                 os.unlink(item_path)
    #                 print(f"Deleted file: {item_path}")
    #             elif os.path.isdir(item_path):
    #                 shutil.rmtree(item_path)
    #                 print(f"Deleted directory: {item_path}")
    #         except Exception as e:
    #             print(f"Failed to delete {item_path}. Reason: {e}")
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        S2_RGB_10_TOKEN,
        L8_RGB_30_TOKEN,
        S2_MS_10_TOKEN,
        HIGH_RGB_05_TOKEN,
        HIGH_RGB_05_TEMP_TOKEN,
        HIGH_RGBI_05,
        S1_VH_10_TOKEN,
        S1_VH_1_TOKEN,
        TREECLASSIFY,
        GROUNDING,
        REFER,
        CLASSIFY,
        IDENTIFY,
        CAPTION,
        CHANGEDET,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader("~/petreloss.conf") if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info("Loading InternVLChatModel...")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
            logger.info("Using flash_attention_2 for InternLM")
        else:
            config.llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
            logger.info("Using flash_attention_2 for LLaMA")
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config
        )
    else:
        logger.info("Loading ViT-6B...")
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config
        )
        logger.info("Loading LLaMA...")
        llm_config = AutoConfig.from_pretrained(
            model_args.llm_path, trust_remote_code=True
        )
        if llm_config.model_type == "internlm2":
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = "flash_attention_2"  # for InternLM
            logger.info("Using flash_attention_2 for InternLM")
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
            logger.info("Using flash_attention_2 for LLaMA")
        llm = model_type.from_pretrained(
            model_args.llm_path,
            torch_dtype=torch.bfloat16,
            config=llm_config,
            trust_remote_code=True,
        )
        logger.info("Building InternVLChatConfig...")
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info("Building InternVLChatModel...")
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info("Finished")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(
        f"model.config.vision_config.image_size: {model.config.vision_config.image_size}"
    )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from "
            f"{model.config.vision_config.image_size} "
            f"to {data_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio**2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    # train_dataset = build_datasets_threaded(
    #     data_args,
    #     tokenizer,
    #     tcs_loader,
    #     model,
    #     group_by_length=training_args.group_by_length,
    #     dynamic_image_size=data_args.dynamic_image_size,
    #     use_thumbnail=data_args.use_thumbnail,
    #     min_dynamic_patch=data_args.min_dynamic_patch,
    #     max_dynamic_patch=data_args.max_dynamic_patch,
    #     normalize_type=data_args.normalize_type
    # )
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(
            r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora
        )
        model.config.use_backbone_lora = model_args.use_backbone_lora
        logger.info(f"model.config.use_backbone_lora: {model.config.use_backbone_lora}")

    if model_args.use_llm_lora:
        model.wrap_llm_lora(
            r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora
        )
        model.config.use_llm_lora = model_args.use_llm_lora
        logger.info(f"model.config.use_llm_lora: {model.config.use_llm_lora}")

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()
    
    # Create a DataLoader for the training dataset, incorporating the sampler
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=training_args.per_device_train_batch_size,
    #     sampler=train_sampler,  # Add the DistributedSampler for distributed training
    #     shuffle=(train_sampler is None),  # Shuffle only if no DistributedSampler is used
    #     pin_memory=True,  # Enable pinned memory for faster data transfer to GPU
    #     num_workers=training_args.dataloader_num_workers
    # )
    # # Initialize the DistributedSampler if distributed training is active
    # if torch.distributed.is_initialized():
    #     train_sampler = DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=concat_pad_data_collator,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()

# # --------------------------------------------------------
# # GEOVLM
# # Copyright (c) 2024 
# # Licensed under The MIT License [see LICENSE for details]
# # --------------------------------------------------------

# import gc
# import json
# import logging
# import math
# import os
# import random
# import sys
# import traceback
# import warnings
# from copy import deepcopy
# from dataclasses import dataclass, field
# from typing import Dict, Optional

# import numpy as np
# import torch
# import torch.distributed as dist
# import transformers
# from earthdial.dist_utils import init_dist
# from earthdial.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
# from earthdial.model.internvl_chat import (InternVisionConfig,
#                                           InternVisionModel,
#                                           InternVLChatConfig,
#                                           InternVLChatModel)
# from earthdial.patch import (concat_pad_data_collator,
#                             replace_llama_rmsnorm_with_fused_rmsnorm,
#                             replace_train_sampler)
# from earthdial.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
#                                       IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
#                                       IMG_START_TOKEN, QUAD_END_TOKEN,
#                                       QUAD_START_TOKEN, REF_END_TOKEN,
#                                       REF_START_TOKEN)
# from earthdial.train.dataset import (ConcatDataset, TCSLoader,
#                                     WeightedConcatDataset, build_transform,
#                                     dynamic_preprocess, preprocess,
#                                     preprocess_internlm, preprocess_mpt,
#                                     preprocess_phi3)
# from earthdial.train.trainer_monkey_patch import replace_create_optimizer
# from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
# from torch.utils.data import Dataset
# from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
#                           HfArgumentParser, Trainer, TrainingArguments,
#                           set_seed)
# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils.logging import (enable_default_handler,
#                                         enable_explicit_format, set_verbosity)

# # Apply necessary patches for the transformers library
# replace_llama_rmsnorm_with_fused_rmsnorm()
# replace_train_sampler()

# # Try to import petrel_client for image loading, fallback to PIL if unavailable
# try:
#     from petrel_client.client import Client
#     from petrel_client.common.config import Config
#     has_tcs_loader = True
# except ImportError as E:
#     print('petrel_client is not installed. Using PIL to load images.')
#     has_tcs_loader = False

# # Set constants for image processing and logging
# IGNORE_INDEX = -100
# Image.MAX_IMAGE_PIXELS = None
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# MaximumDecompressedSize = 1024
# MegaByte = 2 ** 20
# PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

# warnings.filterwarnings('ignore')
# logger = logging.getLogger(__name__)

# os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# @dataclass
# class ModelArguments:
#     """
#     Arguments for specifying model, tokenizer, and configurations.
#     """
#     model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
#     )
#     vision_path: Optional[str] = field(
#         default=None,
#         metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
#     )
#     llm_path: Optional[str] = field(
#         default=None,
#         metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
#     )
#     mlp_path: Optional[str] = field(
#         default=None,
#         metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
#     )
#     freeze_llm: bool = field(
#         default=False,
#         metadata={'help': 'Set to True to freeze the LLM decoder.'},
#     )
#     freeze_backbone: bool = field(
#         default=False,
#         metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
#     )
#     freeze_mlp: bool = field(
#         default=False,
#         metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
#     )
#     unfreeze_vit_layers: int = field(
#         default=0,
#         metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
#     )
#     vision_select_layer: int = field(
#         default=-1,
#         metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
#     )
#     use_backbone_lora: int = field(
#         default=0,
#         metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
#     )
#     use_llm_lora: int = field(
#         default=0,
#         metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
#     )
#     unfreeze_lm_head: bool = field(
#         default=False,
#         metadata={'help': "Set to True to unfreeze the language model's head."},
#     )
#     use_custom_trainer: bool = field(
#         default=False,
#         metadata={'help': 'Set to True to enable the use of a custom trainer.'},
#     )
#     grad_checkpoint: Optional[bool] = field(
#         default=False,
#         metadata={'help': 'Set to True to use gradient checkpointing.'},
#     )
#     drop_path_rate: float = field(
#         default=0.0,
#         metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
#     )
#     ps_version: str = field(
#         default='v2',
#         metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
#                           'Please use `v2` to fix the bug of transposed image.'}
#     )


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments for specifying data input for training and evaluation.
#     """
#     max_seq_length: Optional[int] = field(
#         default=2048,
#         metadata={
#             'help': (
#                 'The maximum total input sequence length after tokenization. Sequences longer '
#                 'than this will be truncated, sequences shorter will be padded.'
#             )
#         },
#     )
#     force_image_size: Optional[int] = field(
#         default=448,
#         metadata={'help': 'Set the desired size for the image. Default is 224.'},
#     )
#     down_sample_ratio: Optional[float] = field(
#         default=0.5,
#         metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
#     )
#     pad2square: Optional[bool] = field(
#         default=False,
#         metadata={'help': 'Pad the image to a square shape if set to True.'},
#     )
#     conv_style: Optional[str] = field(
#         default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
#     )
#     meta_path: Optional[str] = field(
#         default=None,
#         metadata={'help': 'The path of the meta file of datasets.'},
#     )
#     use_data_resampling: Optional[bool] = field(
#         default=False,
#         metadata={'help': 'Set to True to use data resampling.'},
#     )
#     dynamic_image_size: Optional[bool] = field(
#         default=False,
#         metadata={'help': 'Set to True to use dynamic image size.'},
#     )
#     use_thumbnail: Optional[bool] = field(
#         default=False,
#         metadata={'help': 'Set to True to add a thumbnail image.'},
#     )
#     min_dynamic_patch: Optional[int] = field(
#         default=1,
#         metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
#     )
#     max_dynamic_patch: Optional[int] = field(
#         default=12,
#         metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
#     )
#     normalize_type: Optional[str] = field(
#         default='imagenet',
#         metadata={'help': 'The normalize type for the image. Default is imagenet.'},
#     )


# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(
#         self,
#         template_name,
#         meta,
#         tokenizer,
#         tcs_loader,
#         ds_name,
#         num_image_token,
#         image_size=224,
#         is_train=True,
#         pad2square=False,
#         group_by_length=False,
#         dynamic_image_size=False,
#         use_thumbnail=False,
#         min_dynamic_patch=1,
#         max_dynamic_patch=6,
#         min_num_frame=4,  # for video data
#         max_num_frame=12,  # for video data
#         sampling_method='rand',  # for video data
#         repeat_time=1,
#         normalize_type='imagenet',
#         random_seed=0,
#     ):
#         super(LazySupervisedDataset, self).__init__()
#         self.ds_name = ds_name
#         self.tokenizer = tokenizer
#         self.template_name = template_name
#         self.num_image_token = num_image_token
#         logger.info(f'[Dataset] num_image_token: {num_image_token}')
#         logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
#         logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
#         logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

#         self.image_size = image_size
#         self.is_train = is_train
#         self.pad2square = pad2square
#         self.max_num_frame = max_num_frame
#         self.min_num_frame = min_num_frame
#         self.sampling_method = sampling_method

#         logger.info('Formatting inputs...Skip in lazy mode')
#         assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

#         total_ranks = torch.distributed.get_world_size()
#         current_rank = torch.distributed.get_rank()

#         """
#         This section of the code is used to read hundreds of millions of data entries.
#         By using caching and splitting the data according to rank, it ensures fast reading
#         speed and prevents out-of-memory.
#         """
#         # Create a cache directory path
#         basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
#         data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')
#         os.makedirs(data_dir, exist_ok=True)  # Create the cache directory if it does not exist
#         # Create a temporary path for the current rank
#         temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')

#         # Check if the temporary file for the current rank already exists
#         if os.path.exists(temp_path):
#             # If it exists, read the raw data from the file
#             with open(temp_path, 'r') as f:
#                 self.raw_data = f.readlines()
#         else:
#             # If it does not exist, read the raw data from the original annotation file
#             with open(meta['annotation'], 'r') as f:
#                 self.raw_data = f.readlines()

#             # Adjust the raw data based on the repeat_time parameter
#             if repeat_time < 1:
#                 self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
#             else:
#                 self.raw_data = self.raw_data * int(repeat_time)

#             # Calculate the total number of lines and distribute lines to each rank
#             total_lines = len(self.raw_data)
#             logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
#             lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
#             lines_per_rank = max(1, lines_per_rank)

#             # Calculate the start and end line numbers for the current rank
#             start_line = lines_per_rank * current_rank  # Starting line for the current rank
#             end_line = start_line + lines_per_rank  # Ending line for the current rank

#             # Assign the appropriate lines to the current rank
#             self.raw_data = self.raw_data[start_line:end_line]

#             # Write the raw data for the current rank to the temporary file
#             with open(temp_path, 'w') as f:
#                 f.writelines(self.raw_data)

#         self.rng = np.random.default_rng(seed=random_seed)
#         self.rng.shuffle(self.raw_data)

#         gc.collect()
#         self.root = meta['root']
#         self.cached_data_dict = {}
#         self.tcs_loader = tcs_loader
#         self.group_by_length = group_by_length
#         self.dynamic_image_size = dynamic_image_size
#         self.use_thumbnail = use_thumbnail
#         self.min_dynamic_patch = min_dynamic_patch
#         self.max_dynamic_patch = max_dynamic_patch
#         self.normalize_type = normalize_type

#         # If the precomputed length does not exist, roughly estimate the length of
#         # each sample to improve the efficiency of group_by_length.
#         if self.group_by_length:
#             self.conv2length = {}  # Using a dictionary to speed up token length calculation
#             self.length = []
#             for data_item in self.raw_data:
#                 data_item = json.loads(data_item)
#                 if 'length' in data_item:
#                     token_length = data_item['length']  # Use precomputed length if available
#                 else:
#                     # Compute token length using the tokenizer
#                     conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
#                     str_length = len(conversations)
#                     if str_length not in self.conv2length:
#                         token_length = tokenizer(
#                             conversations, return_tensors='pt', padding=False, truncation=False,
#                         ).input_ids.size(1)
#                         self.conv2length[str_length] = token_length + num_image_token * (
#                                     max_dynamic_patch + use_thumbnail)
#                     else:
#                         token_length = self.conv2length[str_length]
#                 self.length.append(token_length)
#         gc.collect()

#     def __len__(self):
#         return len(self.raw_data) * torch.distributed.get_world_size()

#     def get_preprocess_function(self):
#         # Select the appropriate preprocessing function based on the template name
#         if self.template_name == 'Hermes-2':
#             preprocess_function = preprocess_mpt
#         elif self.template_name == 'internlm2-chat':
#             preprocess_function = preprocess_internlm
#         elif self.template_name == 'phi3-chat':
#             preprocess_function = preprocess_phi3
#         else:
#             preprocess_function = preprocess
#         return preprocess_function

#     def load_image(self, image_path):
#         # Load the image using tcs_loader if available, otherwise use PIL
#         if self.tcs_loader is not None and 's3://' in image_path:
#             return self.tcs_loader(image_path)
#         return Image.open(image_path).convert('RGB')

#     def get_image_path(self, image_path):
#         if image_path.startswith('s3://'):  # for ceph
#             image_path = self.root + image_path
#         else:  # for local image
#             image_path = os.path.join(self.root, image_path)
#         return image_path

#     def get_transform(self):
#         # Build transformation function
#         transform = build_transform(is_train=self.is_train, input_size=self.image_size,
#                                     pad2square=self.pad2square, normalize_type=self.normalize_type)
#         return transform

#     def multi_modal_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         # Ensure the first conversation contains an image placeholder
#         if '<image>' not in data_item['conversations'][0]['value']:
#             data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

#         # Merge the image path
#         image_path = self.get_image_path(data_item['image'])

#         # Load the image using tcs_loader if available, otherwise use PIL
#         image = self.load_image(image_path)

#         if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
#             images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
#                                         image_size=self.image_size, use_thumbnail=self.use_thumbnail)
#         else:  # Otherwise, use the original image as a single patch
#             images = [image]

#         # Apply the transformation to each image and stack the results into a tensor
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)

#         # Ensure that there is only one patch if dynamic image size is not enabled
#         num_patches = pixel_values.size(0)
#         if not self.dynamic_image_size:
#             assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
#                                   self.tokenizer, [self.num_image_token * num_patches],
#                                   group_by_length=self.group_by_length, ds_name=self.ds_name)

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def multi_modal_multi_image_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         images, num_tiles = [], []
#         num_image = len(data_item['image'])
#         for image_path in data_item['image']:
#             # Merge the image path
#             image_path = self.get_image_path(image_path)
#             # Load the image using tcs_loader if available, otherwise use PIL
#             image = self.load_image(image_path)
#             if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
#                 image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
#                                            max_num=self.max_dynamic_patch // num_image,
#                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
#                 images += image
#                 num_tiles.append(len(image))
#             else:  # Otherwise, use the original image as a single patch
#                 images.append(image)
#                 num_tiles.append(1)
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)
#         num_patches = pixel_values.size(0)

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
#         ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
#                                   self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
#                                   ds_name=self.ds_name, num_image=num_image)

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def video_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         # Ensure the first conversation contains a video placeholder
#         if '<video>' not in data_item['conversations'][0]['value']:
#             data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

#         # Get the video file path
#         video_file = data_item['video']
#         video_path = os.path.join(self.root, video_file)

#         # Load the video frames using tcs_loader
#         # TODO: Load videos without using tcsloader.
#         image_list = self.tcs_loader(
#             video_path,
#             image_type='video',
#             max_num_frames=self.max_num_frame,
#             min_num_frames=self.min_num_frame,
#             sample=self.sampling_method,
#             clip=data_item.get('clip', None))

#         # Generate special tokens for each video frame
#         special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
#         data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
#             '<video>\n', special_tokens)

#         # Transform each frame image and stack them into a tensor
#         pixel_values = [transform(image) for image in image_list]
#         pixel_values = torch.stack(pixel_values)
#         num_patches = pixel_values.size(0)

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         num_image_tokens = [self.num_image_token] * num_patches
#         ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
#                                   self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
#                                   ds_name=self.ds_name, num_image=num_patches)

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def pure_text_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         # Create a blank white image
#         image = Image.new('RGB', (224, 224), (255, 255, 255))

#         # Dynamically preprocess the image to generate patches
#         images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
#                                     image_size=self.image_size, use_thumbnail=self.use_thumbnail)

#         # Apply the transformation to each image patch and stack them into a tensor
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)
#         num_patches = pixel_values.size(0)

#         # Ensure there is only one patch
#         assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
#                                   self.tokenizer, [self.num_image_token * num_patches], text_only=True,
#                                   group_by_length=self.group_by_length, ds_name=self.ds_name)

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret['input_ids'][0],
#             labels=ret['labels'][0],
#             attention_mask=ret['attention_mask'][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
#         )
#         return ret

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         i = i % len(self.raw_data)
#         while True:
#             try:
#                 data_item = json.loads(self.raw_data[i])
#                 if 'image' in data_item and len(data_item['image']) != 0:
#                     if type(data_item['image']) == list:
#                         ret = self.multi_modal_multi_image_get_item(data_item)
#                     else:
#                         ret = self.multi_modal_get_item(data_item)
#                 elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
#                     ret = self.video_get_item(data_item)
#                 else:
#                     ret = self.pure_text_get_item(data_item)
#                 break
#             except Exception as e:
#                 print(e, self.ds_name, flush=True)
#                 if not isinstance(e, UnidentifiedImageError):
#                     traceback.print_exc()
#                 data_item = json.loads(self.raw_data[i])
#                 if 'image' in data_item:
#                     if type(data_item['image']) == list:
#                         images = [self.root + item for item in data_item['image']]
#                         print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
#                     else:
#                         if data_item['image'].startswith('s3://'):
#                             data_path = self.root + data_item['image']
#                         else:
#                             data_path = os.path.join(self.root, data_item['image'])
#                         print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
#                 elif 'video' in data_item:
#                     data_path = os.path.join(self.root, data_item['video'])
#                     print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
#                 i = random.randint(0, len(self.raw_data) - 1)
#         return ret


# def build_datasets(
#     data_args,
#     tokenizer,
#     tcs_loader,
#     model,
#     group_by_length=False,
#     dynamic_image_size=False,
#     use_thumbnail=False,
#     min_dynamic_patch=1,
#     max_dynamic_patch=12,
#     normalize_type='imagenet',
# ):
#     datasets = []
#     lengths = []
#     ds_collections = json.loads(open(data_args.meta_path).read())
#     for ds_idx, ds_name in enumerate(ds_collections.keys()):
#         repeat_time = ds_collections[ds_name]['repeat_time']
#         if 'max_dynamic_patch' in ds_collections[ds_name]:
#             max_num = ds_collections[ds_name]['max_dynamic_patch']
#             logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
#         else:
#             max_num = max_dynamic_patch
#         dataset = LazySupervisedDataset(
#             data_args.conv_style, ds_collections[ds_name],
#             tokenizer,
#             tcs_loader,
#             ds_name=ds_name,
#             num_image_token=model.num_image_token,
#             image_size=data_args.force_image_size,
#             is_train=ds_collections[ds_name]['data_augment'],
#             pad2square=data_args.pad2square,
#             group_by_length=group_by_length,
#             dynamic_image_size=dynamic_image_size,
#             use_thumbnail=use_thumbnail,
#             min_dynamic_patch=min_dynamic_patch,
#             max_dynamic_patch=max_num,
#             repeat_time=repeat_time,
#             normalize_type=normalize_type,
#             random_seed=ds_idx,
#         )
#         logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
#         datasets.append(dataset)
#         if data_args.use_data_resampling:
#             lengths.append(math.sqrt(len(dataset)))
#         else:
#             lengths.append(len(dataset))
#     if data_args.use_data_resampling:
#         total_length = sum(lengths)
#         weights = [l / total_length for l in lengths]
#         train_dataset = WeightedConcatDataset(datasets, weights)
#     else:
#         train_dataset = ConcatDataset(datasets)
#     return train_dataset


# def main():
#     # Parse input arguments
#     # See all possible arguments in src/transformers/training_args.py
#     # If use DeepSpeed zero3, init_dist must before HfArgumentParser
#     launcher = os.environ.get('LAUNCHER', 'slurm')
#     init_dist(launcher=launcher, backend='nccl')
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
#         # If we pass only one argument to the script, and it's the path to a json file,
#         # let's parse it to get our arguments.
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
#     # information sent is the one passed as arguments along with your Python/PyTorch versions.
#     # send_example_telemetry('InternV-Chat', model_args, data_args)

#     # Setup logging
#     logging.basicConfig(
#         format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#         datefmt='%m/%d/%Y %H:%M:%S',
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     if training_args.should_log:
#         # The default of training_args.log_level is passive, so we set log level at info here to have that default.
#         transformers.utils.logging.set_verbosity_info()

#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     set_verbosity(log_level)
#     enable_default_handler()
#     enable_explicit_format()

#     # Log on each process the small summary:
#     logger.warning(
#         f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
#         + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
#     )
#     logger.info(f'Training/evaluation parameters {training_args}')

#     # Detecting last checkpoint and eventually continue from last checkpoint.
#     last_checkpoint = None
#     if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#         if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
#             raise ValueError(
#                 f'Output directory ({training_args.output_dir}) already exists and is not empty. '
#                 'Use --overwrite_output_dir to overcome.'
#             )
#         elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#             logger.info(
#                 f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
#                 'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
#             )
#     # Set seed before initializing model.
#     set_seed(training_args.seed)

#     # Load pretrained model, tokenizer, and image processor
#     tokenizer_path = model_args.model_name_or_path or model_args.llm_path
#     logger.info(f'Loading Tokenizer: {tokenizer_path}')
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
#     tokenizer.tokenizer_path = tokenizer_path
#     tokenizer.model_max_length = data_args.max_seq_length
#     token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
#                   QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
#                   REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
#     num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
#     img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
#     tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

#     if model_args.model_name_or_path is not None:
#         logger.info('Loading InternVLChatModel...')
#         config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
#         config.vision_config.drop_path_rate = model_args.drop_path_rate
#         if config.llm_config.model_type == 'internlm2':
#             config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
#             logger.info('Using flash_attention_2 for InternLM')
#         else:
#             config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
#             logger.info('Using flash_attention_2 for LLaMA')
#         config.template = data_args.conv_style
#         config.select_layer = model_args.vision_select_layer
#         config.dynamic_image_size = data_args.dynamic_image_size
#         config.use_thumbnail = data_args.use_thumbnail
#         config.ps_version = model_args.ps_version
#         config.min_dynamic_patch = data_args.min_dynamic_patch
#         config.max_dynamic_patch = data_args.max_dynamic_patch
#         model = InternVLChatModel.from_pretrained(
#             model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
#     else:
#         logger.info('Loading ViT-6B...')
#         vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
#         vision_config.drop_path_rate = model_args.drop_path_rate
#         vision_model = InternVisionModel.from_pretrained(
#             model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
#         logger.info('Loading LLaMA...')
#         llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
#         if llm_config.model_type == 'internlm2':
#             model_type = InternLM2ForCausalLM
#             llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
#             logger.info('Using flash_attention_2 for InternLM')
#         else:
#             model_type = AutoModelForCausalLM
#             llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
#             logger.info('Using flash_attention_2 for LLaMA')
#         llm = model_type.from_pretrained(
#             model_args.llm_path, torch_dtype=torch.bfloat16,
#             config=llm_config, trust_remote_code=True)
#         logger.info('Building InternVLChatConfig...')
#         internvl_chat_config = InternVLChatConfig(
#             vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
#             pad2square=data_args.pad2square, template=data_args.conv_style,
#             select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
#             use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
#             min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
#         internvl_chat_config.force_image_size = data_args.force_image_size
#         logger.info('Building InternVLChatModel...')
#         model = InternVLChatModel(internvl_chat_config, vision_model, llm)
#     model.img_context_token_id = img_context_token_id

#     assert model.config.downsample_ratio == data_args.down_sample_ratio

#     if model_args.mlp_path is not None:
#         logger.info('Loading pretrained MLP projector...')
#         state_dict = torch.load(model_args.mlp_path, map_location='cpu')
#         message = model.mlp1.load_state_dict(state_dict)
#         logger.info(message)
#     logger.info('Finished')

#     patch_size = model.config.vision_config.patch_size
#     logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
#     logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
#     logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
#     if model.config.vision_config.image_size != data_args.force_image_size:
#         logger.info(f'Resizing position embedding from '
#                     f'{model.config.vision_config.image_size} '
#                     f'to {data_args.force_image_size}...')
#         model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
#                                                  new_size=data_args.force_image_size,
#                                                  patch_size=patch_size)
#         model.config.vision_config.image_size = data_args.force_image_size
#     model.config.force_image_size = data_args.force_image_size
#     model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

#     if num_new_tokens > 0:
#         model.language_model.resize_token_embeddings(len(tokenizer))
#         output_embeddings = model.language_model.get_output_embeddings().weight.data
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg

#         model.config.llm_config.vocab_size = len(tokenizer)
#         model.language_model.config.vocab_size = len(tokenizer)

#     model.language_model.config.use_cache = False
#     model.vision_model.gradient_checkpointing = True
#     model.vision_model.encoder.gradient_checkpointing = True
#     if model_args.grad_checkpoint:
#         model.language_model._set_gradient_checkpointing()

#     train_dataset = build_datasets(
#         data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
#         dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
#         min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
#         normalize_type=data_args.normalize_type)

#     def _freeze_params(module):
#         for param in module.parameters():
#             param.requires_grad = False

#     if model_args.freeze_backbone:
#         # model.vision_model = model.vision_model.eval()
#         _freeze_params(model.vision_model)

#     if model_args.freeze_llm:
#         model.language_model = model.language_model.eval()
#         _freeze_params(model.language_model)

#     if model_args.unfreeze_lm_head:
#         model.language_model.lm_head.requires_grad = True

#     if model_args.use_backbone_lora:
#         model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
#         model.config.use_backbone_lora = model_args.use_backbone_lora

#     if model_args.use_llm_lora:
#         model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
#         model.config.use_llm_lora = model_args.use_llm_lora

#     if model_args.freeze_mlp:
#         _freeze_params(model.mlp1)

#     if model_args.unfreeze_vit_layers != 0:
#         layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
#         for k, v in layers.named_parameters():
#             logger.info(f'Unfreezing ViT layer: {k}')
#             v.requires_grad = True

#     # print trainable parameters
#     if dist.get_rank() == 0:
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 logger.info(name)

#     # set seed for torch dataloaders
#     set_seed(training_args.seed)

#     # Initialize our Trainer
#     if model_args.use_custom_trainer:
#         replace_create_optimizer()

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset if training_args.do_train else None,
#         eval_dataset=None,
#         tokenizer=tokenizer,
#         data_collator=concat_pad_data_collator
#     )

#     # Training
#     if training_args.do_train:
#         checkpoint = None
#         if training_args.resume_from_checkpoint is not None:
#             checkpoint = training_args.resume_from_checkpoint
#         elif last_checkpoint is not None:
#             checkpoint = last_checkpoint
#         train_result = trainer.train(resume_from_checkpoint=checkpoint)
#         trainer.save_model()  # Saves the tokenizer too for easy upload

#         metrics = train_result.metrics
#         try:
#             metrics['train_samples'] = len(train_dataset)
#         except:
#             metrics['train_samples'] = -1

#         trainer.log_metrics('train', metrics)
#         trainer.save_metrics('train', metrics)
#         trainer.save_state()


# if __name__ == '__main__':
#     main()
