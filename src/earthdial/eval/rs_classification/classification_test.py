import argparse
import itertools
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional
import sys
sys.path.append('./src')
import torch
from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_from_disk
import numpy as np
import warnings
import torch.distributed as dist
from itertools import islice
import cv2

warnings.filterwarnings("once")



ds_collections = {
    'AID': {
        'shard_path': './validation_data/Classification/AID',
        'max_new_tokens': 10,
        'normalization': 'imagenet',
        'pooling': None
    },
    'UCM': {
        'shard_path': './validation_data/Classification/UCM',
        'max_new_tokens': 10,
        'normalization': 'imagenet',
        'pooling': None
    },
    'WHU_19': {
        'shard_path': './validation_data/Classification/WHU_19',
        'max_new_tokens': 10,
        'normalization': 'imagenet',
        'pooling': None
    },
    'BigEarthNet_RGB': {
        'shard_path': './validation_data/Classification/BigEarthNet_RGB/BigEarthNet_test',
        'max_new_tokens': 500,
        'normalization': 'imagenet',
        'pooling': None
    },  
    'rs_LCZ_test': {
        'shard_path': './validation_data/Classification/LCZs_S2/LCZs_S2_test',
        'max_new_tokens': 10,
        'bands':10,
        'normalization':'s2_norm',
        'pooling': 'bilinear'
    },
    'TreeSatAI': {
        'shard_path': './validation_data/Classification/TreeSatAI/TreeSatAI_test',
        'max_new_tokens': 10,
        'bands':4,
        'normalization':'tree_norm',
        'pooling': 'bilinear'
    },
    'BigEarthNet_S2': {
        'shard_path': './validation_data/Classification/BigEarthNet_S2/BigEarthNet_S2_Test',
        'max_new_tokens': 500,
        'bands':12,
        'normalization':'s2_l2a',
        'po,oling': 'average'
    },
    'STARCOP_test': {
        'shard_path': './validation_data/STARCOP_shards/STARCOP_shards/STARCOP_test_yes_or_no',
        'max_new_tokens': 10,
        'bands':4,
        "image_key": "tif_pl,mag1c",
        'normalization':'rgbm_norm',
        'pooling': 'bilinear'
    },
    'UHI_test': {
        'shard_path': './validation_data/UHI_shards/UHI_temperature_landuse_test',
        'max_new_tokens': 50,
        'bands':8,
        'normalization':'l8_norm',
        'pooling': 'bilinear'

    }
    
}



def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, annotations




class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, model, ds_name, shard_path, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, normalize_type="imagenet", pooling='average'):
        
        self.test = load_from_disk(shard_path)
        self.ds_name = ds_name
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

        self.model=model        
        self.transform = build_transform(is_train=False, input_size=input_size,normalize_type=normalize_type)
        self.pooling = pooling
        

    def __len__(self):
        return len(self.test)

    def convert_ms_bands(self, band_images):
        band_images = np.array(band_images, dtype=np.float32)

        if band_images.ndim < 3 or band_images.size == 0:
            raise ValueError("Invalid input: Bands must have at least 3 dimensions and be non-empty.")

        resized_bands = [
            cv2.resize(band, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            if band.shape[:2] != (self.input_size, self.input_size) else band
            for band in band_images
        ]

        return torch.tensor(np.stack(resized_bands, axis=0), dtype=torch.float32)

    def __getitem__(self, idx):

        data = self.test[idx]
        question = data['question']
        annotation = data['groundtruth']


        if self.ds_name in {'rs_LCZ_test', 'BigEarthNet_S2', 'UHI_test'}:
            image = data['tif_ms']
        elif self.ds_name == 'TreeSatAI':
            image = data['rgbi']
        elif self.ds_name == 'STARCOP_test':
            image_keys = ["tif_pl", "mag1c"]  # List of image keys
            image_objects = [np.array(data[key]) for key in image_keys]
            image_objects[0] = np.transpose(image_objects[0], (2, 0, 1))
            image_objects[1] = np.expand_dims(image_objects[1], axis=0) if image_objects[1].ndim == 2 else image_objects[1]
            image = np.concatenate(image_objects, axis=0)
        else:
            image = data['jpg'].convert('RGB')
      

        if self.ds_name in {'AID', 'UCM', 'WHU_19', 'BigEarthNet_RGB'}:
            images = (dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
                    if self.dynamic_image_size else [image])
            pixel_values = torch.stack([self.transform(img) for img in images])
        else:
            if self.ds_name == 'UHI_test':
                image = self.convert_ms_bands(image)
            pixel_values = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)
            pixel_values = torch.stack([self.transform(pixel_value) for pixel_value in pixel_values])
            pixel_values = self.model.sequential_vit_features(pixel_values, self.pooling)


        return {
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
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


def evaluate_chat_model(model):

    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:

        dataset = ClassificationDataset(
            model=model,
            ds_name=ds_name,
            shard_path=ds_collections[ds_name]['shard_path'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            normalize_type=ds_collections[ds_name]['normalization'],
            pooling=ds_collections[ds_name]['pooling']
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
        #for pixel_values, questions, annotations in islice(tqdm(dataloader), 10):
        for pixel_values, questions, annotations in tqdm(dataloader):
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
            answers = [pred]
            
            for question, answer, annotation in zip(questions, answers, annotations):
                outputs.append({
                    'question': question.strip('\"'),
                    'answer': answer,
                    'annotation': annotation.strip('\"')
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            checkpoint_name = '_'.join(args.checkpoint.split('/')[-2:])
            results_file = f'{ds_name}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            with open(results_file, 'w') as outfile:
                for entry in merged_outputs:
                    json.dump(entry, outfile)
                    outfile.write('\n') 
            print('Results saved to {}'.format(results_file))

        torch.distributed.barrier()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/EarthDial_4B_RGB')
    parser.add_argument('--datasets', type=str, default='UCM')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--out-dir', type=str, default='results')
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

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
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

    evaluate_chat_model(model)
