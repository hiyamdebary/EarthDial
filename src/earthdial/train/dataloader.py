# Standard Library Imports
import os
import json
import logging
import numpy as np
import torch
import torch.distributed as dist
import time
import gc
from copy import deepcopy
import random
import traceback

from typing import Dict

# Image Handling Libraries
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project-Specific Imports
from earthdial.train.dataset import (
    build_transform,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3,
)

# Third-Party Libraries
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import ImageFile, UnidentifiedImageError
import cv2

class ShardDataLoader(Dataset):
    """
    A custom dataset class for loading and processing data using the Hugging Face dataset library.
    Designed for supervised fine-tuning with features like dynamic image size, token grouping, 
    and distributed training support.

    Attributes:
        model (object): The model instance to be used during data processing.
        logger (logging.Logger): Logger for tracking dataset-related events.
        template_name (str): The name of the template used for data preprocessing.
        meta (dict): Metadata containing dataset configurations like paths, keys, etc.
        tokenizer (object): Tokenizer instance for encoding text data.
        tcs_loader (object): Custom loader for loading specific datasets.
        ds_name (str): Name of the dataset being loaded.
        num_image_token (int): Number of image tokens used for processing.
        image_size (int): Default size for resizing images.
        is_train (bool): Flag to indicate whether the dataset is for training or evaluation.
        pad2square (bool): Whether to pad images to square dimensions.
        group_by_length (bool): Whether to group samples by token length.
        dynamic_image_size (bool): Enable dynamic resizing of images during preprocessing.
        use_thumbnail (bool): Whether to use thumbnails instead of full images.
        min_dynamic_patch (int): Minimum number of dynamic patches.
        max_dynamic_patch (int): Maximum number of dynamic patches.
        sampling_method (str): Method for sampling data (e.g., "rand").
        repeat_time (int): Number of times to repeat the dataset during training.
        normalize_type (str): Normalization type applied to the images.
        random_seed (int): Seed for random operations to ensure reproducibility.
        raw_data (Dataset): Loaded raw data from disk.
        image_key (str): Key to access images in the dataset.
        conversations_key (str): Key to access conversations in the dataset.
        length (list): Precomputed token lengths for samples.
    """

    def __init__(
        self,
        model,
        logger,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        sampling_method="rand",
        repeat_time=1,
        normalize_type="imagenet",
        random_seed=0,
    ):
        super(ShardDataLoader, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.sampling_method = sampling_method
        self.normalize_type = normalize_type
        self.model = model
        self.logger=logger

        # Distributed training configuration
        total_ranks = torch.distributed.get_world_size()
        current_rank = torch.distributed.get_rank()

        # Load metadata and raw data
        if not os.path.exists(meta["annotation"]):
            raise FileNotFoundError(f"Error: File not found - {meta['annotation']}")
        self.raw_data = load_from_disk(meta["annotation"])
        if len(self.raw_data) == 0:
            logger.info("Error: Raw data is empty.")
        else:
            self.image_key = meta["image_key"]
            self.conversations_key = meta["conversation"]
            logger.info(f"Loaded shard dataset: {self.ds_name} with length: {len(self.raw_data)}")

        if "normalization" in meta:
            self.normalize_type = meta["normalization"]
        if "bands" in meta:
            self.no_bands = meta["bands"]

        # Token length grouping configuration
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        if self.group_by_length:
            self._compute_token_lengths(logger, random_seed)

        gc.collect()

    def _compute_token_lengths(self, logger, random_seed):
        """Compute or retrieve token lengths for efficient data grouping."""
        self.conv2length = {}
        self.length = []
        start_time = time.time()

        for data_item1 in self.raw_data[self.conversations_key]:
            if self.ds_name.strip().startswith("Naip"):
                data_item = json.loads(data_item1)
            else:
                data_item = {"conversations": json.loads(data_item1)}

            if "length" in data_item:
                token_length = data_item["length"]
            else:
                conversations = "\n".join([temp["value"] for temp in data_item["conversations"]])
                str_length = len(conversations)

                if str_length not in self.conv2length:
                    token_length = self.tokenizer(
                        conversations, return_tensors="pt", padding=False, truncation=False
                    ).input_ids.size(1)
                    self.conv2length[str_length] = (
                        token_length + self.num_image_token * (self.max_dynamic_patch + self.use_thumbnail)
                    )
                else:
                    token_length = self.conv2length[str_length]

            self.length.append(token_length)

        total_time = time.time() - start_time
        logger.info(f"Token length computation time for {len(self.raw_data)} samples: {total_time:.2f} seconds")
        gc.collect()

   
    def __len__(self):
        """
        Get the total number of data samples in the raw dataset.

        Returns:
            int: The length of the raw dataset.
        """
        return len(self.raw_data) # * torch.distributed.get_world_size()

    def get_preprocess_function(self):
        """
        Select and return the appropriate preprocessing function based on the template name.

        The method chooses a preprocessing function that aligns with the specified `template_name`
        for the dataset:
            - "Hermes-2" uses `preprocess_mpt`
            - "internlm2-chat" uses `preprocess_internlm`
            - "phi3-chat" uses `preprocess_phi3`
            - Default is `preprocess`

        Returns:
            function: The selected preprocessing function.
        """
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def get_transform(self):
        """
        Build and return the transformation function for image preprocessing.

        The transformation function is built based on the following attributes:
            - `is_train`: Indicates if the dataset is for training or evaluation.
            - `image_size`: The target image size for resizing.
            - `pad2square`: Boolean indicating whether to pad images to a square shape.
            - `normalize_type`: The normalization method to apply (e.g., ImageNet).

        Returns:
            function: A transformation function that processes images based on the given parameters.
        """
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform
    
    def multi_modal_get_item(self, data_item):
        """
        Process a single data item to prepare it for multi-modal model input.

        Args:
            data_item (dict): A dictionary containing the image and conversation data.

        Returns:
            dict: A dictionary containing processed inputs, including tokenized conversations,
                attention masks, image pixel values, and other metadata.
        """
        # Build the transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = (
                "<image>\n" + data_item["conversations"][0]["value"]
            )

        image = data_item["image"]

        # Determine image patches based on dynamic image size or the default behavior
        if self.dynamic_image_size and self.image_key != "tif_ms":
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:
            images = [image]  # Use the original image as a single patch

        # Handle multi-band images
        if self.no_bands != 3:
            if self.image_key == "tif_ms" or str(self.ds_name).strip().startswith("STARCOP"):
                # Handle multi-spectral or special datasets
                pixel_values_ms = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(0)
                pixel_values_norm = [transform(image) for image in pixel_values_ms]
                pixel_values = torch.stack(pixel_values_norm)
            elif self.image_key == "rgbi":
                # Handle RGBI images
                pixel_values_ms = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(0)
                pixel_values_norm = [transform(image) for image in pixel_values_ms]
                pixel_values = torch.stack(pixel_values_norm)
            else:
                # Handle other types of datasets like SAR or NIR or Methane plume
                images = torch.tensor(np.array(images), dtype=torch.float32)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)

            # Extract features using the model's ViT and obtain the embedding space
            pixel_values = self.model.sequential_vit_features(pixel_values, 'bilinear')
            num_patches = pixel_values.size(0)
        else:
            # Apply transformation to the image(s) and stack into a tensor
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            num_patches = pixel_values.size(0)

        # Ensure a single patch for non-dynamic image size
        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate input features
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        # Prepare the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        """
        Process a single data item containing multiple images to prepare inputs for a multi-modal model.

        Args:
            data_item (dict): A dictionary containing image data and associated conversations.

        Returns:
            dict: A dictionary containing processed inputs, including tokenized conversations,
                attention masks, image pixel values, and other metadata.
        """
        # Build the transformation function
        transform = self.get_transform()

        images, num_tiles = [], []  # Initialize containers for processed images and tile counts
        num_image = len(data_item["image"])  # Number of images in the data item

        # Process each image in the data item
        for each_image in data_item["image"]:
            image = each_image
            if self.dynamic_image_size:
                # Dynamically preprocess the image into multiple patches
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))  # Record the number of tiles per image
            else:
                # Use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        # Handle multi-band temporal images
        if self.no_bands != 3:
            images = torch.tensor(np.array(images), dtype=torch.float32)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            self.logger.info("Passing to model Vit ", pixel_values.size())
            # Extract features using the model's ViT and obtain the embedding space
            pixel_values = self.model.sequential_vit_features(pixel_values, 'bilinear')
            self.logger.info("Final ViT ", pixel_values.size())
            num_patches = pixel_values.size(0)
        else:
            # Transform and stack image tensors for RGB inputs
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Prepare token counts for each image based on the number of tiles
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        # Preprocess the conversations and generate input features
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        # Prepare the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)  # Ensure 'i' is within dataset bounds

        while True:
            try:
                # Determine the dataset type and prepare the data item accordingly
                if str(self.ds_name).strip().startswith("Naip"):
                    # NAIP Dataset: Load image and conversation
                    data_item = {
                        "image": self.raw_data[i][self.image_key],
                        "conversations": json.loads(self.raw_data[i][self.conversations_key])["conversations"],
                    }
                elif str(self.ds_name).strip().startswith("Change_SAR"):
                    # Change_SAR Dataset: Handle multiple image keys and process them
                    image_list = self.image_key.split(",")
                    image_objects = [
                        np.transpose(np.array(self.raw_data[i][img_key]), (2, 0, 1))
                        for img_key in image_list
                    ]
                    data_item = {
                        "image": image_objects,
                        "conversations": json.loads(self.raw_data[i][self.conversations_key]),
                    }
                elif str(self.ds_name).strip().startswith("Change"):
                    # Change Dataset: Directly access and store image objects
                    image_list = self.image_key.split(",")
                    image_objects = [self.raw_data[i][img_key] for img_key in image_list]
                    data_item = {
                        "image": image_objects,
                        "conversations": json.loads(self.raw_data[i][self.conversations_key]),
                    }
                elif str(self.ds_name).strip().startswith("STARCOP"):
                    # STARCOP Dataset: Combine images into a single array with 4 channels
                    image_list = self.image_key.split(",")
                    patch_size=512
                    image_objects = [np.array(self.raw_data[i][img_key]) for img_key in image_list]

                    combined_image = np.zeros((self.no_bands, patch_size, patch_size))  # Initialize combined image
                    combined_image[:3, :, :] = np.transpose(image_objects[0], (2, 0, 1))  # RGB
                    if len(image_objects[1].shape) == 2:
                        combined_image[3, :, :] = image_objects[1]
                    else:
                        combined_image[3, :, :] = image_objects[1][0, :, :]
                    data_item = {
                        "image": combined_image,
                        "conversations": json.loads(self.raw_data[i][self.conversations_key]),
                    }
                else:
                    # Default: Load image and conversation
                    data_item = {
                        "image": self.raw_data[i][self.image_key],
                        "conversations": json.loads(self.raw_data[i][self.conversations_key]),
                    }

                # Validate conversations; retry with a random index if missing
                if not data_item["conversations"]:
                    logging.error(f"Empty conversations at index {i} for dataset: {self.ds_name}")
                    i = random.randint(0, len(self.raw_data) - 1)
                    continue

                # Determine the processing function based on the presence and type of 'image' or 'video'
                if "image" in data_item:
                    if isinstance(data_item["image"], list) and str(self.ds_name).strip().startswith("Change"):
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)

                # Exit the loop with the processed data item
                break

            except Exception as e:
                # Handle truncated image or other exceptions
                if "truncated" in str(e):
                    logging.info(f"Truncated image at index {i}, skipping. Dataset: {self.ds_name}")

                # Log and retry on other errors
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()

                logging.info(f"Error loading item {i} from dataset: {self.ds_name}")
                logging.info(str(e))

                # Retry with a random index
                i = random.randint(0, len(self.raw_data) - 1)

        return ret
