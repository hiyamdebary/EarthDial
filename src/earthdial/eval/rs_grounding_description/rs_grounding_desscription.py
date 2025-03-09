import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional
import sys
sys.path.append('./internvl_chat')
import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import json

import nltk

# Download necessary resources if not already available
nltk.download('wordnet')

from datasets import load_from_disk
from itertools import islice

import warnings

warnings.filterwarnings("once")


ds_collections = {
    'HIT_UAV': {
            'shard_path': './validation_shards/grounding_description/HIT_UAV_grounding_test',
            'max_new_tokens': 300,
            'org_prompt': " [grounding] [Inf_thermal_1] <image> \\n Describe the image.",
            'req_prompt': "[grounding] [Inf_thermal_1] <image> \n Describe the image in detail. Detect objects location. Describe objects size. Where objects are located?. Describe relation between objects."
        },
    'NWPU_VHR_10': {
        'shard_path': './validation_shards/grounding_description/NWPU_VHR_10_grounding_test',
        'max_new_tokens': 100,
        'org_prompt': " [grounding] [hr_rgb_0.5] <image> \\n Describe the image.",
        'req_prompt': "[grounding] [hr_rgb_0.5] <image> \n Describe the image in detail. Detect objects location. Describe objects size. Where objects are located?. Describe relation between objects."
    },
    'Swimming_pool_dataset': {
            'shard_path': './validation_shards/grounding_description/Swimming_pool_dataset_grounding_test',
            'max_new_tokens': 100,
            'org_prompt': " [grounding] [hr_rgb_0.5] <image> \\n Describe the image.",
            'req_prompt': "[grounding] [hr_rgb_0.5] <image> \n Describe the image in detail. Detect objects location. Describe objects size. Where objects are located?. Describe relation between objects."
        },
    'UCAS_AOD': {
            'shard_path': './validation_shards/grounding_description/UCAS_AOD_grounding_test',
            'max_new_tokens': 300,
            'org_prompt': " [grounding] [hr_rgb_0.5] <image> \\n Describe the image.",
            'req_prompt': "[grounding] [hr_rgb_0.5] <image> \n Describe the image in detail. Detect objects location. Describe objects size. Where objects are located?. Describe relation between objects."
        },
}

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    groundtruths = [_['groundtruth'] for _ in batches]
    obj_ids = [_['obj_ids'] for _ in batches]
    size_groups = [_['size_group'] for _ in batches]
    obj_locations = [_['obj_location'] for _ in batches]
    obj_names = [_['obj_name'] for _ in batches]

    return pixel_values, questions, groundtruths, obj_ids, size_groups, obj_locations, obj_names



class RSDataset(torch.utils.data.Dataset):

    def __init__(self, ds_name, shard_path, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.test = load_from_disk(shard_path)
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.ds_name = ds_name
        
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        
        data = self.test[idx]
        
            
        image = data['jpg'].convert('RGB')
        question = data['question']
        question = question.replace(ds_collections[self.ds_name]['org_prompt'], ds_collections[self.ds_name]['req_prompt'])
        groundtruth = data['groundtruth']
        obj_ids = data['obj_ids']
        size_group = data['size_group']
        obj_location = data['obj_location']
        obj_name = data['obj_name']
            
            
        
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        
        return {
            'question': question,
            'pixel_values': pixel_values,
            'groundtruth': groundtruth,
            'obj_ids': obj_ids,
            'size_group': size_group,
            'obj_location': obj_location,
            'obj_name': obj_name
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


def evaluate_chat_model():
    
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        

        dataset = RSDataset(
            ds_name=ds_name,
            shard_path=ds_collections[ds_name]['shard_path'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
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
        for _, (pixel_values, questions, groundtruths, obj_ids, size_groups, obj_locations, obj_names) in enumerate(tqdm(dataloader)):
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
                verbose=True
            )
            answers = [pred]
            
            for question, answer, groundtruth, obj_id, size_group, obj_location, obj_name in zip(questions, answers, groundtruths, obj_ids, size_groups, obj_locations, obj_names):
                
                outputs.append({
                    'question': question.strip('\"'),
                    'answer': answer,
                    'groundtruth': groundtruth.strip('\"'),
                    'obj_ids': obj_id.strip('\"'),
                    'size_group': size_group.strip('\"'),
                    'obj_location': obj_location.strip('\"'),
                    'obj_name': obj_name.strip('\"')
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'{ds_name} Evaluation Complete...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            with open(results_file, 'w') as outfile:
                for entry in merged_outputs:
                    json.dump(entry, outfile)
                    outfile.write('\n')  # Write a newline after each JSON object
            print('Results saved to {}'.format(results_file))


        torch.distributed.barrier()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--base_path', type=str, default='./pretrained/')
    parser.add_argument('--datasets', type=str, default='HIT_UAV,NWPU_VHR_10,Swimming_pool_dataset,UCAS_AOD')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print("+++++++++++++++++++++ Executing Region Captioning Script +++++++++++++++++++++")
    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    args.checkpoint = args.base_path + args.checkpoint

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
