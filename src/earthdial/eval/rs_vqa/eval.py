from nltk.tokenize import word_tokenize
import json
import os
import argparse
from colorama import Fore, init
init(autoreset=True)  # Initialize Colorama


def evaluate_f1(reference, candidate):
    reference = reference.strip().lower().replace(" ", "")
    candidate = candidate.strip().lower().replace(" ", "")
    #print(reference, candidate)
    ref_tokens = word_tokenize(reference.strip().lower())
    cand_tokens = word_tokenize(candidate.strip().lower())
    
    common = set(ref_tokens) & set(cand_tokens)
    if not common:
        return 0  # No match at all

    recall = len(common) / len(ref_tokens)
    
    return recall




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='RSVQA_LR')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'src/earthdial/eval/rs_vqa/results/{ds_name}.jsonl'
        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]       

        print(f'{Fore.RED}VQA Scores for {ds_name} dataset')


        f = open(json_path)
        data = [json.loads(line) for line in f.readlines()]

        # Initialize fscore tracking
        vqa_types = ['comp', 'presence', 'rural_urban']

        pred_scores = {vqa_type: 0 for vqa_type in vqa_types}  # Store fscore for each sample
        sample_count = {vqa_type: 0 for vqa_type in vqa_types}    # Count occurrences per vqa_type

        for idx, item in enumerate(data):
            response = item['answer']
            reference = item['annotation']
            vqa_type = item['vqa_type']
            
            if vqa_type in {'comp','presence','rural_urban'}:
                pred_scores[vqa_type] += evaluate_f1(reference, response)
                sample_count[vqa_type] += 1
                

        # Compute final fscore per task type
        for vqa_type in pred_scores:
            if sample_count[vqa_type] > 0:
                pred_scores[vqa_type] /= sample_count[vqa_type]  # Average fscore per task type


        print('-'*40)
        
        for vqa_type, fscore in pred_scores.items():
            print(f'{Fore.RED}{vqa_type} Objects: {fscore:.4f}')
        
        print('-'*40)




