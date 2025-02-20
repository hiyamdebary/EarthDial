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
import time
import gc
from copy import deepcopy
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
import cv2
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from datafusion import DataFusion

class ShardDataLoader_pretrain(Dataset):
    """Dataset for loading data using hugging face data loader for the supervised fine-tuning."""

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
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        sampling_method="rand",  # for video data
        repeat_time=1,
        normalize_type="imagenet",
        random_seed=0,
    ):
        super(ShardDataLoader_pretrain, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(
            f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
        )
        logger.info(f"[Data_loader] Starting with shard dataset: {self.ds_name}")
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.normalize_type=normalize_type
        self.model=model
        # distributed
        total_ranks = torch.distributed.get_world_size()
        current_rank = torch.distributed.get_rank()

        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        #self.raw_data = load_from_disk(meta["annotation"])

        try:
            logger.info(f"Reading dataset: {self.ds_name}")
            raw_data1 = load_from_disk(meta["annotation"])
            # Calculate the total number of lines and distribute lines to each rank
            total_lines = len(raw_data1)
            logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
            lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
            lines_per_rank = max(1, lines_per_rank)

            # Calculate the start and end line numbers for the current rank
            start_line = lines_per_rank * current_rank  # Starting line for the current rank
            end_line = start_line + lines_per_rank  # Ending line for the current rank
            logger.info(f'start_line: {start_line}, end_line: {end_line}')

            # Assign the appropriate lines to the current rank
            self.raw_data = raw_data1.select(range(start_line,end_line))
            logger.info(f"Loaded shard dataset: {self.ds_name} with length: {len(self.raw_data)}")
            # Clear raw_data1 to free memory
            del raw_data1
            gc.collect()  # Force garbage collection to release memory
            logger.info(f"Cleared memory for the shard dataset: {self.ds_name} ")

        except Exception as e:
                logger.info(f"Error in reading datasets {e}")
        if(len(self.raw_data)==0):
            logger.info(f"Error in self raw data")
        else:
            self.image_key = meta["image_key"]
            logger.info(f"Loaded shard dataset: {self.ds_name} with length: {len(self.raw_data)}")
            self.conversations_key = meta["conversation"] 
            self.rng = np.random.default_rng(seed=random_seed)
           # self.raw_data = self.raw_data.shuffle(seed=random_seed)
        if "normalization" in meta:
            self.normalize_type = meta["normalization"]
        if "bands" in meta:
            self.no_bands = meta["bands"]
        if self.image_key == "tif_ms":
            self.normalize_type =meta["normalization"] 
            self.no_bands = meta["bands"] # len(self.raw_data[0]["tif_ms"])
            # self.vit = DataFusion(
            #     img_size=(self.image_size, self.image_size),
            #     patch_size=14,
            #     emb_dim=1024,
            #     num_heads=16,
            #     mlp_dim=3072,
            #     depth=8,
            #     decoder_depth=4,
            #     in_channels=self.no_bands,
            # )
        gc.collect()
    #    self.root = meta["root"]
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}
            # Using a dictionary to speed up token length calculation
            self.length = []

            # Start timing
            start_time = time.time()

            for data_item1 in self.raw_data[self.conversations_key]:
                # Extract conversations from the data item
                if(str(self.ds_name).strip().startswith("Naip")):
                    data_item=json.loads(data_item1)
                else:
                    data_item = {
                        "conversations": json.loads(data_item1),
                    }

                # Check if the length is precomputed
                if "length" in data_item:
                    token_length = data_item["length"]
                else:
                    # Compute token length using the tokenizer
                    conversations = "\n".join(
                        [temp["value"] for temp in data_item["conversations"]]
                    )
                    str_length = len(conversations)

                    # Check if the length for this string has been computed before
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = (
                            token_length
                            + num_image_token * (max_dynamic_patch + use_thumbnail)
                        )
                    else:
                        token_length = self.conv2length[str_length]

                # Append the computed or retrieved token length to the length list
                self.length.append(token_length)

            # Calculate the total time taken
            total_time = time.time() - start_time

            # Log the total time taken
            logger.info(
                f"Total time taken to compute token length  {len(self.raw_data)} samples: {total_time:.2f} seconds"
            )
            gc.collect()
    
   
    def __len__(self):
        return len(self.raw_data)* torch.distributed.get_world_size()

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function
    
    def convert_ms_bands(self, band_images):
        # Initialize a list to hold the resized bands
        resized_bands = []
        band_images = band_images[0]

        for idx, band in enumerate(band_images):
            try:
                # Check if the band is already a NumPy array
                if not isinstance(band, np.ndarray):
                    band = np.array(band)

                # Check if the band is None or empty
                if band is None or band.size == 0:
                    raise ValueError(f"Band {idx} is None or empty.")

                # Check dimensions
                if band.ndim < 2:
                    raise ValueError(
                        f"Band {idx} does not have the correct dimensions. Shape: {band.shape}"
                    )
                # Convert dtype to float32
                band = band.astype(np.float32)  # Convert to float32

                # Print shape before resizing
                #print(f"Processing band {idx}, shape: {band.shape}")

                # Resize if necessary
                if band.shape[0] != self.image_size or band.shape[1] != self.image_size:
                    band = cv2.resize(band, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

                # Append resized band
                resized_bands.append(band)
            except Exception as e:
                print(f"Error processing band {idx}: {e}")

        # Stack and convert to PyTorch tensor
        if len(resized_bands) == 0:
            raise ValueError("No valid bands to stack.")

        stacked_array = np.stack(resized_bands, axis=0)  # Shape: (N, 448, 448)
        result_tensor = torch.tensor(stacked_array, dtype=torch.float32)

        return result_tensor

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform
    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = (
                "<image>\n" + data_item["conversations"][0]["value"]
            )
        image = data_item["image"]

        if self.dynamic_image_size & (not self.image_key == "tif_ms"):
            # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:
            # Otherwise, use the original image as a single patch
            images = [image]
        if self.no_bands != 3:
            if self.image_key == "tif_ms":
                # pixel_values_ms = self.convert_ms_bands(images)
                # # image_ms_tensor = torch.tensor(result_array, dtype=torch.float32)
                # pixel_values_ms = pixel_values_ms.unsqueeze(0)
                # pixel_values_norm = [transform(image) for image in pixel_values_ms]
                # pixel_values= torch.stack(pixel_values_norm)
                pixel_values_ms=torch.tensor(np.array(images),dtype=torch.float32)
                # image_ms_tensor = torch.tensor(pixel_values_ms_1, dtype=torch.float32)
                pixel_values_ms = pixel_values_ms.unsqueeze(0)
                pixel_values_norm = [transform(image) for image in pixel_values_ms]
                pixel_values= torch.stack(pixel_values_norm)
            elif self.image_key == "rgbi":
                pixel_values_ms=torch.tensor(np.array(images),dtype=torch.float32)
                # image_ms_tensor = torch.tensor(pixel_values_ms_1, dtype=torch.float32)
                pixel_values_ms = pixel_values_ms.unsqueeze(0)
                pixel_values_norm = [transform(image) for image in pixel_values_ms]
                pixel_values= torch.stack(pixel_values_norm)
            else:
                #print("Reading datasets............", self.ds_name)
                # pixel_values_ms=torch.tensor(np.array(images),dtype=torch.float32)
                # pixel_values_ms = pixel_values_ms.unsqueeze(0)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
            
            # Apply Vit and get the embedding space..
            pixel_values=self.model.sequential_vit_features(pixel_values,'bilinear')
            num_patches = pixel_values.size(0)
        
        # if self.image_key == "tif_ms":
        #     pixel_values_ms = self.convert_ms_bands(images)
        #     # image_ms_tensor = torch.tensor(result_array, dtype=torch.float32)
        #     pixel_values_ms = pixel_values_ms.unsqueeze(0)
        #     pixel_values_norm = [transform(image) for image in pixel_values_ms]
        #     pixel_values_norm = torch.stack(pixel_values_norm)
        #     # apply Vit and get the embedding space..
        #     pixel_values = self.vit(pixel_values_norm)
        #     #pixel_values = pixel_values.unsqueeze(0)
        #     #print("MS fusion: ", pixel_values.shape)
            
        #     # Ensure that there is only one patch if dynamic image size is not enabled
        #     num_patches = pixel_values_ms.size(0)
        else:
            # Apply the transformation to each image and stack the results into a tensor
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            # Ensure that there is only one patch if dynamic image size is not enabled
            num_patches = pixel_values.size(0)

        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret
    def multi_modal_get_item_old(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = (
                "<image>\n" + data_item["conversations"][0]["value"]
            )

        # # Merge the image path
        # image_path = self.get_image_path(data_item["image"])

        # # Load the image using tcs_loader if available, otherwise use PIL
        # image = self.load_image(image_path)
        image = data_item["image"]

        if (self.dynamic_image_size):  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()
          # Ensure the first conversation contains an image placeholder
        # if "<image>" in data_item["conversations"][0]["value"]:
        #     data_item["conversations"][0]["value"] = ( data_item["conversations"][0]["value"] + "\n <image>")
            
        images, num_tiles = [], []
        num_image = len(data_item["image"])
        for each_image in data_item["image"]:
            # Merge the image path
            # Load the image using tcs_loader if available, otherwise use PIL
            image = each_image
            if (
                self.dynamic_image_size
            ):  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret
   
    def transform_conversations(self, conversations):
        # Ensure conversations is a list of dictionaries
        if not isinstance(conversations, list):
            raise ValueError("Conversations should be a list of dictionaries")

        # Transform each dictionary into the desired format
        transformed_conversations = []
        for i, entry in enumerate(conversations):
            if (
                not isinstance(entry, dict)
                or "from" not in entry
                or "value" not in entry
            ):
                raise ValueError(
                    "Each conversation entry should be a dictionary with 'from' and 'value' keys"
                )

            from_role = entry["from"]  # Use the provided role
            conversation_entry = {"from": from_role, "value": entry["value"]}
            transformed_conversations.append(conversation_entry)
        return transformed_conversations

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                # if self.image_key in self.raw_data[i]:
                #     image = self.raw_data[i][self.image_key]
                # else:
                #     print(f"Key '{self.image_key}' not found in entry {i} for {self.ds_name}. Available keys: {self.raw_data[i].keys()}")
                    
                if(str(self.ds_name).strip().startswith("Naip")):
                        data_item = {
                        "image": self.raw_data[i][self.image_key],
                        "conversations": json.loads(self.raw_data[i][self.conversations_key])["conversations"],
                                }
                elif(str(self.ds_name).strip().startswith("Change")):
                        image_list = self.image_key.split(",")
                        # Access and store each image object dynamically
                        image_objects = [self.raw_data[i][img_key] for img_key in image_list] 
                        data_item = {
                        "image": image_objects,
                        "conversations": json.loads(self.raw_data[i][self.conversations_key])
                        }
                else:
                        data_item = {
                        "image": self.raw_data[i][self.image_key],
                        "conversations": json.loads(
                            self.raw_data[i][self.conversations_key]
                        ),
                                    }
                    
                if 'image' in data_item :
                    if (type(data_item['image']) == list) & (str(self.ds_name).strip().startswith("Change")):
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            # except Exception as e:
            #     print(e, self.ds_name, flush=True)
            except Exception as e:
                if "truncated" in str(e):
                    logging.info(f"Truncated image at index {i}, skipping... the dataset is: {self.ds_name}")
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                logging.info(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                if(str(self.ds_name).strip().startswith("Naip")):
                        data_item = {
                        "image": self.raw_data[i][self.image_key],
                        "conversations": json.loads(self.raw_data[i][self.conversations_key])["conversations"],
                                }
                else:
                    data_item = {
                    "image": self.raw_data[i][self.image_key],
                    "conversations": json.loads(
                        self.raw_data[i][self.conversations_key]
                    ),
                }
                logging.info(f'Failed to load image from id: {i}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret
        
                # if not isinstance(e, UnidentifiedImageError):
                #     traceback.print_exc()

# class ShardDataLoader(Dataset):
#     """Dataset for loading data using hugging face data loader for the supervised fine-tuning."""

#     def __init__(
#         self,
#         logger,
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
#         sampling_method="rand",  # for video data
#         repeat_time=1,
#         normalize_type="imagenet",
#         random_seed=0,
#     ):
#         super(ShardDataLoader, self).__init__()
#         self.ds_name = ds_name
#         self.tokenizer = tokenizer
#         self.template_name = template_name
#         self.num_image_token = num_image_token
#         logger.info(f"[Dataset] num_image_token: {num_image_token}")
#         logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
#         logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
#         logger.info(
#             f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
#         )
#         logger.info(f"[Data_loader] Starting with shard dataset: {self.ds_name}")
#         self.image_size = image_size
#         self.is_train = is_train
#         self.pad2square = pad2square
#         self.max_num_frame = max_num_frame
#         self.min_num_frame = min_num_frame
#         self.sampling_method = sampling_method
#         # distributed
#         total_ranks = torch.distributed.get_world_size()
#         current_rank = torch.distributed.get_rank()

#         """
#         This section of the code is used to read hundreds of millions of data entries.
#         By using caching and splitting the data according to rank, it ensures fast reading
#         speed and prevents out-of-memory.
#         """
#         if str(self.ds_name).strip().startswith("111"):
#                 file_path = meta["file_path"]
#                 no_shards = meta["no_shards"]
                
#                 # Read the shard file paths
#                 with open(file_path, "r") as file:
#                     shard_files = [line.strip() for line in file.readlines()]
                
#                 # Calculate the indices to split the shard files into three chunks
#                 total_shards = len(shard_files)
#                 first_split = total_shards // 3
#                 second_split = 2 * (total_shards // 3)
                
#                 # Split the shard files into three parts: first, second, and third chunk
#                 first_chunk = shard_files[:first_split]
#                 second_chunk = shard_files[first_split:second_split]
#                 third_chunk = shard_files[second_split:]
                
#                 # Select shards based on the value of 'no_shards'
#                 if str(no_shards).strip() == "first_chunk":
#                     shard_files = first_chunk[0:50]
#                 elif str(no_shards).strip() == "second_chunk":
#                     shard_files = second_chunk[0:50]
#                 elif str(no_shards).strip() == "third_chunk":
#                     shard_files = third_chunk[0:50]

#                 logging.info(f"Read NAIP shard txt file. Loading {len(shard_files)} shards...")

#                 # Load datasets sequentially (without multiprocessing)
#                 datasets = []
#                 for shard in shard_files:
#                     dataset = load_from_disk(shard)
#                     #self.load_shard(shard)  # Sequentially load each shard
#                     datasets.append(dataset)

#                 if datasets:
#                     try:
#                         logging.info("Concatenating NAIP datasets...")
#                         raw_data1 = concatenate_datasets(datasets)
#                         # Calculate the total number of lines and distribute lines to each rank
#                         total_lines = len(raw_data1)
#                         logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
#                         lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
#                         lines_per_rank = max(1, lines_per_rank)

#                         # Calculate the start and end line numbers for the current rank
#                         start_line = lines_per_rank * current_rank  # Starting line for the current rank
#                         end_line = start_line + lines_per_rank  # Ending line for the current rank
#                         logger.info(f'start_line: {start_line}, end_line: {end_line}')

#                         # Assign the appropriate lines to the current rank
#                         self.raw_data = raw_data1.select(range(start_line,end_line))

#                         logging.info("NAIP Datasets concatenated successfully!")
#                     except Exception as e:
#                         logging.error(f"Error during NAIP datasets concatenation: {e}")
#                 else:
#                     logging.warning("No datasets were loaded successfully, skipping concatenation.")
#         else:
#             try:
#                 logger.info(f"Reading dataset: {self.ds_name}")
#                 raw_data1 = load_from_disk(meta["annotation"])
#                 # Calculate the total number of lines and distribute lines to each rank
#                 total_lines = len(raw_data1)
#                 logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
#                 lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
#                 lines_per_rank = max(1, lines_per_rank)

#                 # Calculate the start and end line numbers for the current rank
#                 start_line = lines_per_rank * current_rank  # Starting line for the current rank
#                 end_line = start_line + lines_per_rank  # Ending line for the current rank
#                 logger.info(f'start_line: {start_line}, end_line: {end_line}')

#                 # Assign the appropriate lines to the current rank
#                 self.raw_data = raw_data1.select(range(start_line,end_line))
#                 logger.info(f"Loaded shard dataset: {self.ds_name} with length: {len(self.raw_data)}")
#             except Exception as e:
#                     logger.info(f"Error in reading datasets {e}")

#         self.image_key = meta["image_key"]
#         self.conversations_key = meta["conversation"]
                
#         self.rng = np.random.default_rng(seed=random_seed)
#         self.raw_data = self.raw_data.shuffle(seed=random_seed)
        
#     #     gc.collect()
#     #    self.root = meta["root"]
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
#             self.conv2length = {}
#             # Using a dictionary to speed up token length calculation
#             self.length = []

#             # Start timing
#             start_time = time.time()

#             for data_item1 in self.raw_data[self.conversations_key]:
#                 # Extract conversations from the data item
#                 if(str(self.ds_name).strip().startswith("Naip")):
#                     data_item=json.loads(data_item1)
#                 else:
#                     data_item = {
#                         "conversations": json.loads(data_item1),
#                     }

#                 # Check if the length is precomputed
#                 if "length" in data_item:
#                     token_length = data_item["length"]
#                 else:
#                     # Compute token length using the tokenizer
#                     conversations = "\n".join(
#                         [temp["value"] for temp in data_item["conversations"]]
#                     )
#                     str_length = len(conversations)

#                     # Check if the length for this string has been computed before
#                     if str_length not in self.conv2length:
#                         token_length = tokenizer(
#                             conversations,
#                             return_tensors="pt",
#                             padding=False,
#                             truncation=False,
#                         ).input_ids.size(1)
#                         self.conv2length[str_length] = (
#                             token_length
#                             + num_image_token * (max_dynamic_patch + use_thumbnail)
#                         )
#                     else:
#                         token_length = self.conv2length[str_length]

#                 # Append the computed or retrieved token length to the length list
#                 self.length.append(token_length)

#             # Calculate the total time taken
#             total_time = time.time() - start_time

#             # Log the total time taken
#             logger.info(
#                 f"Total time taken to compute token length  {len(self.raw_data)} samples: {total_time:.2f} seconds"
#             )
#             gc.collect()
    
   
#     def __len__(self):
#         return len(self.raw_data) * torch.distributed.get_world_size()

#     def get_preprocess_function(self):
#         # Select the appropriate preprocessing function based on the template name
#         if self.template_name == "Hermes-2":
#             preprocess_function = preprocess_mpt
#         elif self.template_name == "internlm2-chat":
#             preprocess_function = preprocess_internlm
#         elif self.template_name == "phi3-chat":
#             preprocess_function = preprocess_phi3
#         else:
#             preprocess_function = preprocess
#         return preprocess_function


#     def get_transform(self):
#         # Build transformation function
#         transform = build_transform(
#             is_train=self.is_train,
#             input_size=self.image_size,
#             pad2square=self.pad2square,
#             normalize_type=self.normalize_type,
#         )
#         return transform

#     def multi_modal_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         # Ensure the first conversation contains an image placeholder
#         if "<image>" not in data_item["conversations"][0]["value"]:
#             data_item["conversations"][0]["value"] = (
#                 "<image>\n" + data_item["conversations"][0]["value"]
#             )

#         # # Merge the image path
#         # image_path = self.get_image_path(data_item["image"])

#         # # Load the image using tcs_loader if available, otherwise use PIL
#         # image = self.load_image(image_path)
#         image = data_item["image"]

#         if (self.dynamic_image_size):  # If dynamic image size is enabled, preprocess the image dynamically
#             images = dynamic_preprocess(
#                 image,
#                 min_num=self.min_dynamic_patch,
#                 max_num=self.max_dynamic_patch,
#                 image_size=self.image_size,
#                 use_thumbnail=self.use_thumbnail,
#             )
#         else:  # Otherwise, use the original image as a single patch
#             images = [image]

#         # Apply the transformation to each image and stack the results into a tensor
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)

#         # Ensure that there is only one patch if dynamic image size is not enabled
#         num_patches = pixel_values.size(0)
#         if not self.dynamic_image_size:
#             assert (
#                 num_patches == 1
#             ), f"The number of patches should be 1, but got {num_patches}."

#         # Select the appropriate preprocessing function based on the template name
#         preprocess_function = self.get_preprocess_function()

#         # Preprocess the conversations and generate the return dictionary
#         ret = preprocess_function(
#             self.template_name,
#             [deepcopy(data_item["conversations"])],
#             self.tokenizer,
#             [self.num_image_token * num_patches],
#             group_by_length=self.group_by_length,
#             ds_name=self.ds_name,
#         )

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret["input_ids"][0],
#             labels=ret["labels"][0],
#             attention_mask=ret["attention_mask"][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
#         )
#         return ret

#     def multi_modal_multi_image_get_item(self, data_item):
#         # Build transformation function
#         transform = self.get_transform()

#         images, num_tiles = [], []
#         num_image = len(data_item["image"])
#         for image_path in data_item["image"]:
#             # Merge the image path
#             image_path = self.get_image_path(image_path)
#             # Load the image using tcs_loader if available, otherwise use PIL
#             image = self.load_image(image_path)
#             if (
#                 self.dynamic_image_size
#             ):  # If dynamic image size is enabled, preprocess the image dynamically
#                 image = dynamic_preprocess(
#                     image,
#                     min_num=self.min_dynamic_patch,
#                     max_num=self.max_dynamic_patch // num_image,
#                     image_size=self.image_size,
#                     use_thumbnail=self.use_thumbnail,
#                 )
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
#         ret = preprocess_function(
#             self.template_name,
#             [deepcopy(data_item["conversations"])],
#             self.tokenizer,
#             num_image_tokens,
#             group_by_length=self.group_by_length,
#             ds_name=self.ds_name,
#             num_image=num_image,
#         )

#         # Create the final return dictionary
#         ret = dict(
#             input_ids=ret["input_ids"][0],
#             labels=ret["labels"][0],
#             attention_mask=ret["attention_mask"][0],
#             pixel_values=pixel_values,
#             image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
#         )
#         return ret

   
#     def transform_conversations(self, conversations):
#         # Ensure conversations is a list of dictionaries
#         if not isinstance(conversations, list):
#             raise ValueError("Conversations should be a list of dictionaries")

#         # Transform each dictionary into the desired format
#         transformed_conversations = []
#         for i, entry in enumerate(conversations):
#             if (
#                 not isinstance(entry, dict)
#                 or "from" not in entry
#                 or "value" not in entry
#             ):
#                 raise ValueError(
#                     "Each conversation entry should be a dictionary with 'from' and 'value' keys"
#                 )

#             from_role = entry["from"]  # Use the provided role
#             conversation_entry = {"from": from_role, "value": entry["value"]}
#             transformed_conversations.append(conversation_entry)
#         return transformed_conversations

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         i = i % len(self.raw_data)
#         while True:
#             try:
#                 # Extract and parse conversations data on-the-fly
#                 # raw_conversations = self.raw_data[i].get("conversations_internlm", "{}")
#                 if(str(self.ds_name).strip().startswith("Naip")):
#                         data_item=json.loads(data_item)
#                         data_item = {
#                         # "image": self.raw_data[i]["png"],
#                         "image": self.raw_data[i][self.image_key],
#                         "conversations": json.loads(self.raw_data[i][self.conversations_key])["conversations"],
#                                 }
#                 else:
#                     data_item = {
#                     # "image": self.raw_data[i]["png"],
#                     "image": self.raw_data[i][self.image_key],
#                     "conversations": json.loads(
#                         self.raw_data[i][self.conversations_key]
#                     ),
#                 }
#                 if "image" in data_item:
#                     if type(data_item["image"]) == list:
#                         ret = self.multi_modal_multi_image_get_item(data_item)
#                     else:
#                         ret = self.multi_modal_get_item(data_item)
#                 else:
#                     ret = self.pure_text_get_item(data_item)
#                 break
#             except Exception as e:
#                 print(e, self.ds_name, flush=True)
#                 if not isinstance(e, UnidentifiedImageError):
#                     traceback.print_exc()
#                 # data_item = json.loads(self.raw_data[i])
#                 # if "image" in data_item:
#                 #     if type(data_item["image"]) == list:
#                 #         images = [self.root + item for item in data_item["image"]]
#                 #         print(
#                 #             f"Failed to load image: {images}, the dataset is: {self.ds_name}"
#                 #         )
#                 #     else:
#                 #         if data_item["image"].startswith("s3://"):
#                 #             data_path = self.root + data_item["image"]
#                 #         else:
#                 #             data_path = os.path.join(self.root, data_item["image"])
#                 #         print(
#                 #             f"Failed to load image: {data_path}, the dataset is: {self.ds_name}"
#                 #         )
#                 # i = random.randint(0, len(self.raw_data) - 1)
#         return ret

# # class LazySupervisedDataset_Vela(Dataset):
# #     """Dataset for loading data using hugging face data loader for the supervised fine-tuning."""

# #     def __init__(
# #         self,
# #         template_name,
# #         meta,
# #         tokenizer,
# #         tcs_loader,
# #         ds_name,
# #         num_image_token,
# #         image_size=224,
# #         is_train=True,
# #         pad2square=False,
# #         group_by_length=False,
# #         dynamic_image_size=False,
# #         use_thumbnail=False,
# #         min_dynamic_patch=1,
# #         max_dynamic_patch=6,
# #         min_num_frame=4,  # for video data
# #         max_num_frame=12,  # for video data
# #         sampling_method="rand",  # for video data
# #         repeat_time=1,
# #         normalize_type="imagenet",
# #         random_seed=0,
# #     ):
# #         super(LazySupervisedDataset_Vela, self).__init__()
# #         self.ds_name = ds_name
# #         self.tokenizer = tokenizer
# #         self.template_name = template_name
# #         self.num_image_token = num_image_token
# #         logger.info(f"[Dataset] num_image_token: {num_image_token}")
# #         logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
# #         logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
# #         logger.info(
# #             f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
# #         )
# #         logger.info(f"[Data_loader] Starting with shard dataset: {self.ds_name}")
# #         self.image_size = image_size
# #         self.is_train = is_train
# #         self.pad2square = pad2square
# #         self.max_num_frame = max_num_frame
# #         self.min_num_frame = min_num_frame
# #         self.sampling_method = sampling_method
# #         # distributed
# #         total_ranks = torch.distributed.get_world_size()
# #         current_rank = torch.distributed.get_rank()

# #         """
# #         This section of the code is used to read hundreds of millions of data entries.
# #         By using caching and splitting the data according to rank, it ensures fast reading
# #         speed and prevents out-of-memory.
# #         """
#  # try:
#             #     logger.info(f"Reading dataset: {self.ds_name}")

#             #     # Cache directory path for storing temp shard files
#             #     basename = os.path.basename(meta['annotation'])  # Modify to match file naming
#             #     data_dir = os.path.join(os.path.dirname(meta['cache']), f'{basename}_temp')
#             #     os.makedirs(data_dir, exist_ok=True)  # Create cache directory if it does not exist

#             #     # Path to temp shard for the current rank
#             #     temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.arrow')
#             #     logger.info(f"Creating tmp: {temp_path}")

#             #     if os.path.exists(temp_path):
#             #         # If the shard file exists, load the dataset from the cached .arrow file
#             #         logger.info(f"Loading cached shard dataset for rank {current_rank} from {temp_path}")
#             #         self.raw_data = load_from_disk(temp_path)
#             #     else:
#             #         # Load the original shard dataset if cache doesn't exist
#             #         self.raw_data = load_from_disk(meta["annotation"])

#             #         # Adjust the dataset based on the repeat_time parameter (if applicable)
#             #         # if repeat_time < 1:
#             #         #     num_rows = int(len(self.raw_data) * repeat_time)
#             #         #     self.raw_data = self.raw_data.select(range(num_rows))
#             #         # else:
#             #         #     self.raw_data = self.raw_data.select(range(len(self.raw_data) * int(repeat_time)))

#             #         # Calculate the total number of rows and distribute rows to each rank
#             #         total_lines = len(self.raw_data)
#             #         logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
#             #         lines_per_rank = total_lines // total_ranks
#             #         lines_per_rank = max(1, lines_per_rank)

#             #         # Calculate the start and end row indices for the current rank
#             #         start_line = lines_per_rank * current_rank
#             #         end_line = start_line + lines_per_rank
#             #         logger.info(f'start_line: {start_line}, end_line: {end_line}')

#             #         # Assign the appropriate rows to the current rank
#             #         self.raw_data = self.raw_data.select(range(start_line, end_line))

#             #         # Save the raw data for the current rank as a .arrow file
#             #         logger.info(f"Saving shard dataset for rank {current_rank} to {temp_path}")
#             #         self.raw_data.save_to_disk(temp_path)

#             #     # Shuffle the dataset if necessary
#             #     self.rng = np.random.default_rng(seed=random_seed)
#             #    # self.raw_data = self.raw_data.shuffle(seed=random_seed)

#             # except Exception as e:
#             #     logger.error(f"Error reading dataset: {e}")

#             # Perform memory cleanup if needed
       
# #         # Create a cache directory path
# #         # basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
# #         # data_dir = os.path.join(os.path.dirname(meta['annotation']), f'{basename}_temp')
# #         # os.makedirs(data_dir, exist_ok=True)  # Create the cache directory if it does not exist
# #         # # Create a temporary path for the current rank
# #         # temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')

# #         if str(self.ds_name).strip().startswith("NAIP"):
# #             file_path = meta["file_path"]
# #             no_shards = meta["no_shards"]
            
# #             # Read the shard file paths
# #             with open(file_path, "r") as file:
# #                 shard_files = [line.strip() for line in file.readlines()]
            
# #             # Calculate the indices to split the shard files into three chunks
# #             total_shards = len(shard_files)
# #             first_split = total_shards // 3
# #             second_split = 2 * (total_shards // 3)
            
# #             # Split the shard files into three parts: first, second, and third chunk
# #             first_chunk = shard_files[:first_split]
# #             second_chunk = shard_files[first_split:second_split]
# #             third_chunk = shard_files[second_split:]
            
# #             # Select shards based on the value of 'no_shards'
# #             if str(no_shards).strip() == "first_chunk":
# #                 shard_files = first_chunk[:100]
# #             elif str(no_shards).strip() == "second_chunk":
# #                 shard_files = second_chunk[:100]
# #             elif str(no_shards).strip() == "third_chunk":
# #                 shard_files = third_chunk[:100]

# #             logging.info(f"Read NAIP shard txt file. Loading {len(shard_files)} shards...")

# #             # Load datasets sequentially (without multiprocessing)
# #             datasets = []
# #             for shard in shard_files:
# #                 dataset = load_from_disk(shard)
# #                 #self.load_shard(shard)  # Sequentially load each shard
# #                 datasets.append(dataset)

# #             if datasets:
# #                 try:
# #                     logging.info("Concatenating NAIP datasets...")
# #                     self.raw_data = concatenate_datasets(datasets)
# #                     # Calculate the total number of lines and distribute lines to each rank
# #                     total_lines = len(self.raw_data)
# #                     logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
# #                     lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
# #                     lines_per_rank = max(1, lines_per_rank)

# #                     # Calculate the start and end line numbers for the current rank
# #                     start_line = lines_per_rank * current_rank  # Starting line for the current rank
# #                     end_line = start_line + lines_per_rank  # Ending line for the current rank
# #                     logger.info(f'start_line: {start_line}, end_line: {end_line}')

# #                     # Assign the appropriate lines to the current rank
# #                     self.raw_data = self.raw_data[start_line:end_line]

# #                     logging.info("NAIP Datasets concatenated successfully!")
# #                 except Exception as e:
# #                     logging.error(f"Error during NAIP datasets concatenation: {e}")
# #             else:
# #                 logging.warning("No datasets were loaded successfully, skipping concatenation.")

# #         else:
# #             try:
# #                 logger.info(f"Reading dataset: {self.ds_name}")
# #                 self.raw_data = load_from_disk(meta["annotation"])
# #                 # Calculate the total number of lines and distribute lines to each rank
# #                 total_lines = len(self.raw_data)
# #                 logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
# #                 lines_per_rank = total_lines // total_ranks  # Number of lines each rank should process
# #                 lines_per_rank = max(1, lines_per_rank)

# #                 # Calculate the start and end line numbers for the current rank
# #                 start_line = lines_per_rank * current_rank  # Starting line for the current rank
# #                 end_line = start_line + lines_per_rank  # Ending line for the current rank
# #                 logger.info(f'start_line: {start_line}, end_line: {end_line}')

# #                 # Assign the appropriate lines to the current rank
# #                 self.raw_data = self.raw_data[start_line:end_line]

# #                 gc.collect()
# #             except Exception as e:
# #                     logger.info(f"Error in reading datasets {e}")
# #         if(len(self.raw_data)==0):
# #             logger.info(f"Error Reading dataset and skipping this for training: {self.ds_name}")
# #         else:
# #             logger.info(f"Loaded shard dataset: {self.ds_name} with length: {len(self.raw_data)}")
# #             self.image_key = meta["image_key"]
# #             self.conversations_key = meta["conversation"]
        
# #             self.rng = np.random.default_rng(seed=random_seed)
# #             #self.raw_data = self.raw_data.shuffle(seed=random_seed)
# #             gc.collect()
# #             self.root = meta["root"]
# #             self.cached_data_dict = {}
# #             self.tcs_loader = tcs_loader
# #             self.group_by_length = group_by_length
# #             self.dynamic_image_size = dynamic_image_size
# #             self.use_thumbnail = use_thumbnail
# #             self.min_dynamic_patch = min_dynamic_patch
# #             self.max_dynamic_patch = max_dynamic_patch
# #             self.normalize_type = normalize_type

# #             # If the precomputed length does not exist, roughly estimate the length of
# #             # each sample to improve the efficiency of group_by_length.
# #             if self.group_by_length:
# #                 self.conv2length = {}
# #                 # Using a dictionary to speed up token length calculation
# #                 self.length = []

# #                 # Start timing
# #                 start_time = time.time()

# #                 for data_item1 in self.raw_data[self.conversations_key]:
# #                     # Extract conversations from the data item
# #                     if(str(self.ds_name).strip().startswith("NAIP")):
# #                         data_item=json.loads(data_item1)
# #                     else:
# #                         data_item = {
# #                             "conversations": json.loads(data_item1),
# #                         }

# #                     # Check if the length is precomputed
# #                     if "length" in data_item:
# #                         token_length = data_item["length"]
# #                     else:
# #                         # Compute token length using the tokenizer
# #                         conversations = "\n".join(
# #                             [temp["value"] for temp in data_item["conversations"]]
# #                         )
# #                         str_length = len(conversations)

# #                         # Check if the length for this string has been computed before
# #                         if str_length not in self.conv2length:
# #                             token_length = tokenizer(
# #                                 conversations,
# #                                 return_tensors="pt",
# #                                 padding=False,
# #                                 truncation=False,
# #                             ).input_ids.size(1)
# #                             self.conv2length[str_length] = (
# #                                 token_length
# #                                 + num_image_token * (max_dynamic_patch + use_thumbnail)
# #                             )
# #                         else:
# #                             token_length = self.conv2length[str_length]

# #                     # Append the computed or retrieved token length to the length list
# #                     self.length.append(token_length)

# #                 # Calculate the total time taken
# #                 total_time = time.time() - start_time

# #                 # Log the total time taken
# #                 logger.info(
# #                     f"Total time taken to compute token length  {len(self.raw_data)} samples: {total_time:.2f} seconds"
# #                 )
# #                 gc.collect()
    
# #     def load_shard(self,shard):
# #         try:
# #             return load_from_disk(shard)
# #         except Exception as e:
# #             logging.warning(f"Failed to load dataset from {shard}: {e}")
# #             return None
# #     def __len__(self):
# #         return len(self.raw_data) * torch.distributed.get_world_size()

# #     def get_preprocess_function(self):
# #         # Select the appropriate preprocessing function based on the template name
# #         if self.template_name == "Hermes-2":
# #             preprocess_function = preprocess_mpt
# #         elif self.template_name == "internlm2-chat":
# #             preprocess_function = preprocess_internlm
# #         elif self.template_name == "phi3-chat":
# #             preprocess_function = preprocess_phi3
# #         else:
# #             preprocess_function = preprocess
# #         return preprocess_function

# #     def load_image(self, image_path):
# #         # Load the image using tcs_loader if available, otherwise use PIL
# #         if self.tcs_loader is not None and "s3://" in image_path:
# #             return self.tcs_loader(image_path)
# #             # Two conditions
# #             # 1) RGB check
# #             # 2) Grayscale (SAR) check
# #             # 3) Temporal check (shape check ())
# #             # 4) Multispectral check
# #         return Image.open(image_path).convert("RGB")

# #     def get_image_path(self, image_path):
# #         if image_path.startswith("s3://"):  # for ceph
# #             image_path = self.root + image_path
# #         else:  # for local image
# #             image_path = os.path.join(self.root, image_path)
# #         return image_path

# #     def get_transform(self):
# #         # Build transformation function
# #         transform = build_transform(
# #             is_train=self.is_train,
# #             input_size=self.image_size,
# #             pad2square=self.pad2square,
# #             normalize_type=self.normalize_type,
# #         )
# #         return transform

# #     def multi_modal_get_item(self, data_item):
# #         # Build transformation function
# #         transform = self.get_transform()

# #         # Ensure the first conversation contains an image placeholder
# #         if "<image>" not in data_item["conversations"][0]["value"]:
# #             data_item["conversations"][0]["value"] = (
# #                 "<image>\n" + data_item["conversations"][0]["value"]
# #             )

# #         # # Merge the image path
# #         # image_path = self.get_image_path(data_item["image"])

# #         # # Load the image using tcs_loader if available, otherwise use PIL
# #         # image = self.load_image(image_path)
# #         image = data_item["image"]

# #         if (
# #             self.dynamic_image_size
# #         ):  # If dynamic image size is enabled, preprocess the image dynamically
# #             images = dynamic_preprocess(
# #                 image,
# #                 min_num=self.min_dynamic_patch,
# #                 max_num=self.max_dynamic_patch,
# #                 image_size=self.image_size,
# #                 use_thumbnail=self.use_thumbnail,
# #             )
# #         else:  # Otherwise, use the original image as a single patch
# #             images = [image]

# #         # Apply the transformation to each image and stack the results into a tensor
# #         pixel_values = [transform(image) for image in images]
# #         pixel_values = torch.stack(pixel_values)

# #         # Ensure that there is only one patch if dynamic image size is not enabled
# #         num_patches = pixel_values.size(0)
# #         if not self.dynamic_image_size:
# #             assert (
# #                 num_patches == 1
# #             ), f"The number of patches should be 1, but got {num_patches}."

# #         # Select the appropriate preprocessing function based on the template name
# #         preprocess_function = self.get_preprocess_function()

# #         # Preprocess the conversations and generate the return dictionary
# #         ret = preprocess_function(
# #             self.template_name,
# #             [deepcopy(data_item["conversations"])],
# #             self.tokenizer,
# #             [self.num_image_token * num_patches],
# #             group_by_length=self.group_by_length,
# #             ds_name=self.ds_name,
# #         )

# #         # Create the final return dictionary
# #         ret = dict(
# #             input_ids=ret["input_ids"][0],
# #             labels=ret["labels"][0],
# #             attention_mask=ret["attention_mask"][0],
# #             pixel_values=pixel_values,
# #             image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
# #         )
# #         return ret

# #     def multi_modal_multi_image_get_item(self, data_item):
# #         # Build transformation function
# #         transform = self.get_transform()

# #         images, num_tiles = [], []
# #         num_image = len(data_item["image"])
# #         for image_path in data_item["image"]:
# #             # Merge the image path
# #             image_path = self.get_image_path(image_path)
# #             # Load the image using tcs_loader if available, otherwise use PIL
# #             image = self.load_image(image_path)
# #             if (
# #                 self.dynamic_image_size
# #             ):  # If dynamic image size is enabled, preprocess the image dynamically
# #                 image = dynamic_preprocess(
# #                     image,
# #                     min_num=self.min_dynamic_patch,
# #                     max_num=self.max_dynamic_patch // num_image,
# #                     image_size=self.image_size,
# #                     use_thumbnail=self.use_thumbnail,
# #                 )
# #                 images += image
# #                 num_tiles.append(len(image))
# #             else:  # Otherwise, use the original image as a single patch
# #                 images.append(image)
# #                 num_tiles.append(1)
# #         pixel_values = [transform(image) for image in images]
# #         pixel_values = torch.stack(pixel_values)
# #         num_patches = pixel_values.size(0)

# #         # Select the appropriate preprocessing function based on the template name
# #         preprocess_function = self.get_preprocess_function()

# #         # Preprocess the conversations and generate the return dictionary
# #         num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
# #         ret = preprocess_function(
# #             self.template_name,
# #             [deepcopy(data_item["conversations"])],
# #             self.tokenizer,
# #             num_image_tokens,
# #             group_by_length=self.group_by_length,
# #             ds_name=self.ds_name,
# #             num_image=num_image,
# #         )

# #         # Create the final return dictionary
# #         ret = dict(
# #             input_ids=ret["input_ids"][0],
# #             labels=ret["labels"][0],
# #             attention_mask=ret["attention_mask"][0],
# #             pixel_values=pixel_values,
# #             image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
# #         )
# #         return ret

   
# #     def transform_conversations(self, conversations):
# #         # Ensure conversations is a list of dictionaries
# #         if not isinstance(conversations, list):
# #             raise ValueError("Conversations should be a list of dictionaries")

# #         # Transform each dictionary into the desired format
# #         transformed_conversations = []
# #         for i, entry in enumerate(conversations):
# #             if (
# #                 not isinstance(entry, dict)
# #                 or "from" not in entry
# #                 or "value" not in entry
# #             ):
# #                 raise ValueError(
# #                     "Each conversation entry should be a dictionary with 'from' and 'value' keys"
# #                 )

# #             from_role = entry["from"]  # Use the provided role
# #             conversation_entry = {"from": from_role, "value": entry["value"]}
# #             transformed_conversations.append(conversation_entry)
# #         return transformed_conversations

# #     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
# #         i = i % len(self.raw_data)
# #         while True:
# #             try:
# #                 # Extract and parse conversations data on-the-fly
# #                 # raw_conversations = self.raw_data[i].get("conversations_internlm", "{}")
# #                 if(str(self.ds_name).strip().startswith("NAIP")):
# #                         data_item=json.loads(data_item)
# #                         data_item = {
# #                         # "image": self.raw_data[i]["png"],
# #                         "image": self.raw_data[i][self.image_key],
# #                         "conversations": json.loads(self.raw_data[i][self.conversations_key])["conversations"],
# #                                 }
# #                 else:
# #                     data_item = {
# #                     # "image": self.raw_data[i]["png"],
# #                     "image": self.raw_data[i][self.image_key],
# #                     "conversations": json.loads(
# #                         self.raw_data[i][self.conversations_key]
# #                     ),
# #                 }
                
# #                 # data_item = json.loads(self.raw_data[i][self.conversations_key])
# #                 # raw_conversations = self.raw_data[i].get(self.conversations_key, "{}")

# #                 # # Convert JSON string to a dictionary
# #                 # try:
# #                 #     conversations_dict = json.loads(raw_conversations)
# #                 # except json.JSONDecodeError:
# #                 #     raise ValueError(
# #                 #         "Error decoding JSON from 'conversations_internlm'"
# #                 #     )

# #                 # # Extract and transform the conversations for NAIP
# #                 # # conversations = conversations_dict.get("conversations", [])

# #                 # transformed_conversations = self.transform_conversations(
# #                 #     conversations_dict
# #                 # )
# #                 # data_item = {
# #                 #     # "image": self.raw_data[i]["png"],
# #                 #     "image": self.raw_data[i][self.image_key],
# #                 #     "conversations": transformed_conversations,
# #                 # }
# #                 # data_item = json.loads(self.raw_data[i])
# #                 # cos_data = self.raw_data[i]
# #                 # data_item["image"] = cos_data["png"]
# #                 # data_item["conversations"] = cos_data["conversations_internlm"]
# #                 #                ret = self.multi_modal_get_item(data_item)
# #                 # if "image" in data_item and len(data_item["image"]) != 0:

# #                 if "image" in data_item:
# #                     if type(data_item["image"]) == list:
# #                         ret = self.multi_modal_multi_image_get_item(data_item)
# #                     else:
# #                         ret = self.multi_modal_get_item(data_item)
# #                 elif (
# #                     "video" in data_item
# #                     and data_item["video"] is not None
# #                     and data_item["video"] != ""
# #                 ):
# #                     ret = self.video_get_item(data_item)
# #                 else:
# #                     ret = self.pure_text_get_item(data_item)
# #                 break
# #             except Exception as e:
# #                 print(e, self.ds_name, flush=True)
# #                 if not isinstance(e, UnidentifiedImageError):
# #                     traceback.print_exc()
# #                 data_item = json.loads(self.raw_data[i])
# #                 if "image" in data_item:
# #                     if type(data_item["image"]) == list:
# #                         images = [self.root + item for item in data_item["image"]]
# #                         print(
# #                             f"Failed to load image: {images}, the dataset is: {self.ds_name}"
# #                         )
# #                     else:
# #                         if data_item["image"].startswith("s3://"):
# #                             data_path = self.root + data_item["image"]
# #                         else:
# #                             data_path = os.path.join(self.root, data_item["image"])
# #                         print(
# #                             f"Failed to load image: {data_path}, the dataset is: {self.ds_name}"
# #                         )
# #                 elif "video" in data_item:
# #                     data_path = os.path.join(self.root, data_item["video"])
# #                     print(
# #                         f"Failed to load video: {data_path}, the dataset is: {self.ds_name}"
# #                     )
# #                 i = random.randint(0, len(self.raw_data) - 1)
# #         return ret

# # def load_single_dataset(ds_info):
# #     ds_name, ds_metadata, tokenizer, tcs_loader, model, data_args, group_by_length, dynamic_image_size, use_thumbnail, min_dynamic_patch, max_dynamic_patch, normalize_type = ds_info
# #     repeat_time = ds_metadata["repeat_time"]
# #     max_num = ds_metadata.get("max_dynamic_patch", max_dynamic_patch)
# #     dataset = LazySupervisedDataset_Vela(
# #         data_args.conv_style,
# #         ds_metadata,
# #         tokenizer,
# #         tcs_loader,
# #         ds_name=ds_name,
# #         num_image_token=model.num_image_token,
# #         image_size=data_args.force_image_size,
# #         is_train=ds_metadata["data_augment"],
# #         pad2square=data_args.pad2square,
# #         group_by_length=group_by_length,
# #         dynamic_image_size=dynamic_image_size,
# #         use_thumbnail=use_thumbnail,
# #         min_dynamic_patch=min_dynamic_patch,
# #         max_dynamic_patch=max_num,
# #         repeat_time=repeat_time,
# #         normalize_type=normalize_type,
# #         random_seed=0
# #     )
# #     return dataset

# # def build_datasets_in_parallel(data_args, tokenizer, tcs_loader, model, group_by_length=False, dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1, max_dynamic_patch=12, normalize_type="imagenet"):
# #     datasets = []
# #     ds_collections = json.loads(open(data_args.meta_path).read())
    
# #     with concurrent.futures.ThreadPoolExecutor() as executor:
# #         futures = []
# #         for ds_idx, ds_name in enumerate(ds_collections.keys()):
# #             ds_metadata = ds_collections[ds_name]
# #             futures.append(executor.submit(load_single_dataset, (ds_name, ds_metadata, tokenizer, tcs_loader, model, data_args, group_by_length, dynamic_image_size, use_thumbnail, min_dynamic_patch, max_dynamic_patch, normalize_type)))
        
# #         for future in concurrent.futures.as_completed(futures):
# #             datasets.append(future.result())
    
# #     train_dataset = ConcatDataset(datasets)
    
#  #   return train_dataset