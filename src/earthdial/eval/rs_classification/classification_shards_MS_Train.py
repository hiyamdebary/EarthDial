import argparse
import itertools
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import random
import subprocess
import time
from functools import partial
from typing import Optional
import sys
# Adjust path based on script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Base path to add: {base_dir}")
path_to_other_project = os.path.join(base_dir, "../../../")
absolute_path = os.path.abspath(path_to_other_project)
if not os.path.exists(absolute_path):
    print(f"Path does not exist: {absolute_path}")
# Add the directory to the system path
sys.path.append(absolute_path)
import torch
from geovlm.model.internvl_chat import InternVLChatModel
from geovlm.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from geovlm.train.datafusion import DataFusion
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk

# Download the punkt resource if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download the punkt_tab resource if not already available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
import cv2
import numpy as np

import json
from datasets import load_from_disk
import torch.distributed as dist
#os.environ['CUDA_VISIBLE_DEVICE']= [4,5]
#"/l/users/fahad.khan/akshay/mbzuai_ibm/cache_hf/cache_hf_new"
#os.environ['TRANSFORMERS_CACHE']="/l/users/fahad.khan/akshay/mbzuai_ibm/cache_hf/cache_hf_new"


ds_collections = {
    'rs_bigearthnet_test': {
        'test': '/share/data/drive_2/remote_sensing/training_shards/BigEarthNet_S2/BigEarthNet_train',
        'max_new_tokens': 50,
        'bands':13,
        'normalization':'s2_l2a'
    }
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]

    return pixel_values, questions, question_ids

def evaluate_f1(reference, candidate):
    reference = reference.strip().lower().replace(" ", "")
    candidate = candidate.strip().lower().replace(" ", "")
    print(reference, candidate)
    ref_tokens = word_tokenize(reference.strip().lower())
    cand_tokens = word_tokenize(candidate.strip().lower())
    
    common = set(ref_tokens) & set(cand_tokens)
    if not common:
        return 0  # No match at all

    recall = len(common) / len(ref_tokens)
    
    return recall

class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, few_shot, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6,normalize_type="imagenet",no_bands=3):
        self.test = load_from_disk(test)
        print(f"Data loaded from the {test} and legnth {len(self.test)}")
        self.no_bands=no_bands
        self.image_size=image_size
        # self.vit = DataFusion(
        #     img_size=(input_size, input_size),
        #     patch_size=14,
        #     emb_dim=1024,
        #     num_heads=16,
        #     mlp_dim=3072,
        #     depth=8,
        #     decoder_depth=4,
        #     in_channels=self.no_bands,
        # )
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        print("Dynamic image size loader",self.dynamic_image_size)
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size,normalize_type=normalize_type)

    def __len__(self):
        return len(self.test)
    def convert_ms_bands(self, band_images):
        # Initialize a list to hold the resized bands
        resized_bands = []
        #band_images = band_images[0]
        band_images = np.array(band_images)

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
    def __getitem__(self, idx):
        data = self.test[idx]

        images = data['tif_ms']
        question = data['conversations']
        question_id=data['__key__']
        # if self.dynamic_image_size:
        #     images = dynamic_preprocess(image, image_size=self.input_size,
        #                                 use_thumbnail=self.use_thumbnail,
        #                                 max_num=self.max_num)
        # else:
        #    images = [image]
        pixel_values_ms = self.convert_ms_bands(images)
        pixel_values_ms = pixel_values_ms.unsqueeze(0)
        pixel_values_norm = [self.transform(image) for image in pixel_values_ms]
        pixel_values= torch.stack(pixel_values_norm)
        #pixel_values = self.vit(pixel_values_norm)

        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        local_rank, rank, world_size=init_distributed_mode()

        self._size = int(size)
        assert size > 0
        self._rank = rank
        self._world_size = world_size
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():

    base_prompt = 'Answer in one word or a short phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    bigearthnet_ms=''
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'rs_ucmerced_test' in ds_name or 'rs_aid_test' in ds_name:
            input_prompt = ai2d_prompt
            print("Base prompt", input_prompt)
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        elif 'rs_bigearthnet_test' in ds_name:
            input_prompt = bigearthnet_ms
        else:
            input_prompt = base_prompt

        norm =ds_collections[ds_name]["normalization"] 
        bands = ds_collections[ds_name]["bands"]

        dataset = VQADataset(
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            normalize_type=norm,
            no_bands=bands
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, question_ids) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                verbose=False
            )
            predictions = [pred]
          
            for question, question_id, predictions in zip(questions, question_ids, predictions):
                if ds_name in ['rs_bigearthnet_test']:
                    predictions=str(predictions).replace(r'\"', '').replace('"', '')
                    # annotation=str(annotation).replace(r'\"', '').replace('"', '')
                    # annotation = annotation.replace('"', '').replace("'", '')


                    print("Actual Annotation:", question)
                    print("Predictions Annotation:",predictions)
            
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'predictions': predictions,
                    })
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            # Open the file in write mode and save each output as a separate JSON line
            with open(results_file, 'w') as f:
                for output in merged_outputs:
                    f.write(json.dumps(output) + '\n')
            #json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))


            recall = 0
            count = 0
            for idx, item in enumerate(merged_outputs):
                response = item['predictions']
                reference = item['annotation']
                
                score = evaluate_f1(reference, response)
                count = count + 1
                recall = recall + score
                
            recall = recall/count
            print(ds_name, recall)
            summaries.append([args.checkpoint, ds_name, recall])


        torch.distributed.barrier()
    out_path = str(args.checkpoint).split('/')[-1]
    # Create the full path for the output file
    output_file_path = os.path.join(str(args.out_dir), f'{out_path}.txt')

    # Open the file in append mode
    with open(output_file_path, 'a') as writer:
        print(f"Writing results to file {output_file_path}")
        # Write each summary to the file
        for summary in summaries:
            print(summary)
            writer.write(f'{summary}\n')
    # out_path = '_'.join(args.checkpoint.split('/')[-1:])
    # writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    # print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    # for summary in summaries:
    #     print(summary)
    #     writer.write(f'{summary}\n')
    # writer.close()


def init_distributed_mode():
    # Specify GPUs 4, 5, 6, 7
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    
    # Determine the backend (nccl for GPUs, gloo for CPUs)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    # Set up default addresses for the master node
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')  # Ensure this port is free or adjust if needed

    # Get or define other necessary environment variables
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    # Initialize the process group if world_size > 1
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank
        )

        # Set the local device for the current process based on restricted GPUs
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"Distributed mode initialized: rank {rank}/{world_size} on device {local_rank} with backend {backend}")
    else:
        print("Running in single-process mode.")
    return local_rank, rank, world_size
# Call the function to initialize the distributed environment
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/share/data/drive_2/remote_sensing/InternVL/pretrained/4B_Full_4Nov_MS_MLP_LLM')
    parser.add_argument('--datasets', type=str, default='rs_bigearthnet_test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=18)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--out-dir', type=str, default='/share/data/drive_2/remote_sensing/InternVL/pretrained/4B_Full_4Nov_MS_MLP_LLM/predictions/')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'
    # Call the function to initialize the distributed environment

    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )

    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
