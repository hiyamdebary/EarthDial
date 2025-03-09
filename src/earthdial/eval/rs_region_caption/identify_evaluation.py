from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
import json
import os


def calculate_rouge_scores(reference_text, candidate_text):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_text, candidate_text)
    
    # Extract ROUGE-1 and ROUGE-L scores
    rouge1 = scores['rouge1']
    rougel = scores['rougeL']
    
    return rouge1, rougel



def calculate_meteor(reference, hypothesis):
    """
    Calculate the METEOR score between a reference sentence and a hypothesis sentence.

    :param reference: A string or a list of strings representing the reference sentence(s)
    :param hypothesis: A string representing the hypothesis (predicted) sentence
    :return: The METEOR score
    """
    # Tokenize both reference and hypothesis
    ref_tokenized = reference.split()  # Tokenizing by splitting on spaces
    hypothesis_tokenized = hypothesis.split()    
    score = meteor_score([ref_tokenized], hypothesis_tokenized)

    return score


shard_path = './urban_tree_crown_detection.jsonl'

f = open(shard_path)
data = [json.loads(line) for line in f.readlines()]


rouge1_precision, rouge1_recall, rouge1_fscore = 0, 0, 0
rougeL_precision, rougeL_recall, rougeL_fscore = 0, 0, 0
M_score = 0

count = 0
for idx, item in enumerate(data):

    response = item['answer']
    reference = item['annotation']
    
    rouge1, rougel = calculate_rouge_scores(reference, response)

    rouge1_precision = rouge1_precision + rouge1.precision
    rouge1_recall = rouge1_recall + rouge1.recall
    rouge1_fscore = rouge1_fscore + rouge1.fmeasure

    rougeL_precision = rougeL_precision + rougel.precision
    rougeL_recall = rougeL_recall + rougel.recall
    rougeL_fscore = rougeL_fscore + rougel.fmeasure

    M_score = M_score + calculate_meteor(reference, response)
    count = count + 1


rouge1_precision = rouge1_precision/count
rouge1_recall = rouge1_recall/count
rouge1_fscore = rouge1_fscore/count

rougeL_precision = rougeL_precision/count
rougeL_recall = rougeL_recall/count
rougeL_fscore = rougeL_fscore/count

M_score = M_score/count

print(f"ROUGE-1: Precision: {rouge1_precision:.4f}, Recall: {rouge1_recall:.4f}, F1 Score: {rouge1_fscore:.4f}")
print(f"ROUGE-L: Precision: {rougeL_precision:.4f}, Recall: {rougeL_recall:.4f}, F1 Score: {rougeL_fscore:.4f}")
print(f"METEOR Score: {M_score:.4f}")

