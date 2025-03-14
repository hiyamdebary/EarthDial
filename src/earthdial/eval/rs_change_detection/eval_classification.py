from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
import json
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
    parser.add_argument('--datasets', type=str, default='FMoW')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'src/earthdial/eval/rs_change_detection/results/{ds_name}.jsonl'
        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]

        recall = 0
        count = 0
        for idx, item in enumerate(data):
            response = item['answer']
            reference = item['caption0']
            
            score = evaluate_f1(reference, response)
            count = count + 1
            recall = recall + score
            
        recall = recall/count
        print(f'{Fore.RED}Classification accuracy for {ds_name} dataset: {recall}')




