import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import sys
sys.path.append('./src')
import torch
from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

import torch.distributed as dist
from itertools import islice

from datasets import load_from_disk



ds_collections = {
    'Dubai_CC': {
        'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/ChangeDetection_Shards/DUBAICC/Dubai_ChangeDetection_test',
        'max_new_tokens': 100,
    },
    'LEVIR_MCI': {
        'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/ChangeDetection_Shards/LEVIRCC/LEVIR_ChangeDetection_test',
        'max_new_tokens': 100,
    },
    'MUDS': {
        'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/ChangeDetection_Shards/MUDS/MUDS_ChangeDetection_test',
        'max_new_tokens': 100,
    },
    'SYSU': {
        'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/SYSU',
        'max_new_tokens': 100,
    },
    'FMoW': {
        'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/FMoW_RGB_Valid',
        'max_new_tokens': 100,
    },
    # Classification task    
    'xBD_testset_1': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_testset_1',
            'max_new_tokens': 10,
            'org_prompt': "[changedet] [hr_rgb_temp_0.5] <image> <image> \\n Analyze the images to identify the type of disaster that occurred. Options:  volcano, fire, earthquake, flood, tsunami, wind.",
            'req_prompt': "[changedet] [hr_rgb_temp_0.5] <image> <image> \\n Analyze the images to identify the type of disaster from one of the following classes. Classes:  volcano, fire, earthquake, flood, tsunami, wind. Answer in single word."
        },
    'xBD_testset_2': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_testset_2',
            'max_new_tokens': 10,
            'org_prompt': " [hr_rgb_temp_0.5] <image>  <image> \\n",
            'req_prompt': "[hr_rgb_temp_0.5] <image>  <image> \n"
        },
    'xBD_testset_3': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_testset_3',
            'max_new_tokens': 10,
            'org_prompt': " [hr_rgb_temp_0.5] <image> <image> \\n",
            'req_prompt': "[hr_rgb_temp_0.5] <image>  <image> \n"
        },
    #Region Classification
    'xBD_reg_cls_testset_1': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_reg_cls_testset_1',
            'max_new_tokens': 10,
            'org_prompt': "classify the level of damage. Classes: no-damage, destroyed, minor-damage, major-damage.",
            'req_prompt': "classify the level of damage in one of the below classes. Classes: no-damage, destroyed, minor-damage, major-damage."
        },
    'xBD_reg_cls_testset_2': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_reg_cls_testset_2',
            'max_new_tokens': 10,
            'org_prompt': "[identify] [hr_rgb_temp_0.5] <image> \\n",
            'req_prompt': "[identify] [hr_rgb_temp_0.5] <image> \n"
        },
    #Image Captioning
    'xBD_image_captioning': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_test_image_captioning',
            'max_new_tokens': 50,
            'org_prompt': "[grounding] [hr_rgb_temp_0.5] <image> <image> \\n",
            'req_prompt': "[grounding] [hr_rgb_temp_0.5] <image> <image> \n"
        },
    #Object Detection: Detect objects same type
    'xBD_object_detection': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_object_detection',
            'max_new_tokens': 100,
            'org_prompt': "[grounding] [hr_rgb_temp_0.5] <image> <image> \\n",
            'req_prompt': "[grounding] [hr_rgb_temp_0.5] <image> <image> \n"
        },
    #Referred Object Detection: Detect referred objects
    'xBD_referred_object_detection': {
            'shard_path': '/share/data/drive_2/remote_sensing/validation_data/Temporal/Disaster_Shards/xBD/test/xBD_referred_object_detection',
            'max_new_tokens': 50,
            'org_prompt': "[refer] [hr_rgb_temp_0.5] <image> <image> \\n",
            'req_prompt': "[refer] [hr_rgb_temp_0.5] <image> <image> \n"
        },
}



def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    caption0 = [_['caption0'] for _ in batches]
    caption1 = [_['caption1'] for _ in batches]
    caption2 = [_['caption2'] for _ in batches]
    caption3 = [_['caption3'] for _ in batches]
    caption4 = [_['caption4'] for _ in batches]

    return pixel_values, questions, caption0, caption1, caption2, caption3, caption4


class CD_Dataset(torch.utils.data.Dataset):

    def __init__(self, ds_name, shard_path, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.test = load_from_disk(shard_path)
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num        
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.ds_name = ds_name

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        
        
        if self.ds_name == 'MUDS' or self.ds_name == 'FMoW':
            image_1 = data['jpg_A'].convert('RGB')
            image_2 = data['jpg_B'].convert('RGB')
            image_3 = data['jpg_C'].convert('RGB')
            image_4 = data['jpg_D'].convert('RGB')
            images = [image_1, image_2, image_3, image_4]
        else:
            image_1 = data['jpg_A'].convert('RGB')
            image_2 = data['jpg_B'].convert('RGB')
            images = [image_1, image_2]        
        
        
        merge_image, num_tiles = [], []
        for image in images:
            if self.dynamic_image_size:
                images = dynamic_preprocess(image, image_size=self.input_size,
                                            use_thumbnail=self.use_thumbnail,
                                            max_num=self.max_num)
                merge_image += images
                num_tiles.append(len(images))
            else:
                merge_image.append(image)
                num_tiles.append(1)

        pixel_values = [self.transform(image) for image in merge_image]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
            
        if self.ds_name in {'FMoW', 'SYSU', 'xBD_testset_1', 'xBD_testset_2', 'xBD_testset_3', 
                     'xBD_image_captioning', 'xBD_object_detection', 'xBD_referred_object_detection', 
                     'xBD_reg_cls_testset_1', 'xBD_reg_cls_testset_2'}:
            question = data['question']
            caption0 = data['groundtruth']
            caption1 = ''
            caption2 = ''
            caption3 = ''
            caption4 = ''
        else:
            question = data['question']
            caption0 = data['groundtruth0']
            caption1 = data['groundtruth1']
            caption2 = data['groundtruth2']
            caption3 = data['groundtruth3']
            caption4 = data['groundtruth4']

        #question = question.replace(ds_collections[self.ds_name]['org_prompt'], ds_collections[self.ds_name]['req_prompt'])
        #question = 'Locate all large <ref>buildings</ref> in the post-disaster image. Give me bounding boxes of the detected large buildings in [xmin, ymin, xmax, ymax, angle] format.'
        
        if self.ds_name in {'xBD_testset_2', 'xBD_testset_3', 'xBD_reg_cls_testset_1', 'xBD_reg_cls_testset_2'}:
            question = question + 'Answer in single word.'
        
    
            
        return {
            'question': question,
            'pixel_values': pixel_values,
            'caption0': caption0,
            'caption1': caption1,
            'caption2': caption2,
            'caption3': caption3,
            'caption4': caption4
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
        

        dataset = CD_Dataset(
            ds_name = ds_name,
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
        for pixel_values, questions, captions0, captions1, captions2, captions3, captions4 in islice(tqdm(dataloader), 5):
        #for pixel_values, questions, captions0, captions1, captions2, captions3, captions4 in tqdm(dataloader):
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
            #print(pred)
            
            for question, answer, caption0, caption1, caption2, caption3, caption4 in zip(questions, answers, captions0, captions1, captions2, captions3, captions4):
                outputs.append({
                        'question': question.strip('\"'),
                        'answer': answer,
                        'caption0': caption0.strip('\"'),
                        'caption1': caption1.strip('\"'),
                        'caption2': caption2.strip('\"'),
                        'caption3': caption3.strip('\"'),
                        'caption4': caption4.strip('\"')
                    })
                

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            with open(results_file, 'w') as outfile:
                for entry in merged_outputs:
                    json.dump(entry, outfile)
                    outfile.write('\n')  # Write a newline after each JSON object
            #json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

        torch.distributed.barrier()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/EarthDial_4B_RGB')
    parser.add_argument('--datasets', type=str, default='xBD_test1')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    print('Dynamic Imaging', args.dynamic)

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

    evaluate_chat_model()
