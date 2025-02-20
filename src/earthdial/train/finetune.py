# --------------------------------------------------------
# EarthDial
# Copyright (c) 2024 
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard Library Imports
import os
import json  
import logging  
import math 
import sys  
import warnings  

# Third-Party Library Imports
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers 
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

# Dataclass and Typing Imports
from dataclasses import dataclass, field 
from typing import Optional  

# Environment Variable Setup
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'  # Set memory trim threshold
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # True parallelism in tokenizers

# Add project-specific paths to `sys.path`
base_dir = os.path.dirname(os.path.abspath(__file__))
path_to_other_project = os.path.join(base_dir, "../../")
absolute_path = os.path.abspath(path_to_other_project)
if not os.path.exists(absolute_path):
    print(f"Path does not exist: {absolute_path}")
sys.path.append(absolute_path)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=r".*Keyword arguments.*not recognized.*")
warnings.filterwarnings("ignore")

# Project-Specific Imports
from earthdial.dist_utils import init_dist  # Distributed training utilities
from earthdial.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM  # Model definitions

# Vision and Chat model configurations
from earthdial.model.internvl_chat import (  
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)
# Custom patches for the training pipeline
from earthdial.patch import (  
    concat_pad_data_collator,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_train_sampler,
)
# Model-specific constants
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
    UHI,
    L8_MS_30,
    HYPER_RGB_3,
    S1_VH_TEMP_10,
    MB_TOKEN_START,
    MB_TOKEN_END,
)
from earthdial.train.dataset import (  # Dataset management utilities
    ConcatDataset,
    TCSLoader,
    WeightedConcatDataset,
)
from earthdial.train.trainer_monkey_patch import replace_create_optimizer  # Custom optimizer patch
from dataloader import ShardDataLoader  # Shard-based data loading utility

# Replace default behavior with custom patches
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Logging Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set default logging level to INFO

# Additional Configuration
has_tcs_loader = False


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
        dataset = ShardDataLoader(
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
        CHANGEDET,UHI,L8_MS_30,HYPER_RGB_3,S1_VH_TEMP_10,MB_TOKEN_START,MB_TOKEN_END
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
    mp.set_start_method('spawn', force=True)
    main()
