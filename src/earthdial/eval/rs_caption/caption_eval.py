from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
import json
import os


def calculate_rouge(hypothesis, references):
    """
    Calculate ROUGE score for a hypothesis against multiple references.

    Parameters:
    hypothesis (str): The generated sentence to evaluate.
    references (list of str): List of reference sentences.

    Returns:
    dict: The average ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) across all references.
    """
    # Tokenize the hypothesis
    hypothesis_tokens = " ".join(word_tokenize(hypothesis))

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE scores for each reference and take the average
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref in references:
        ref_tokens = " ".join(word_tokenize(ref))
        score = scorer.score(ref_tokens, hypothesis_tokens)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure

    # Average the scores over the number of references
    for key in scores:
        scores[key] /= len(references)

    return scores['rouge1'], scores['rougeL']



from nltk.translate.meteor_score import meteor_score

def calculate_meteor(hypothesis, references):
    """
    Calculate METEOR score for a hypothesis against multiple references.

    Parameters:
    hypothesis (str): The generated sentence to evaluate.
    references (list of str): List of reference sentences.

    Returns:
    float: The METEOR score for the hypothesis against the references.
    """
    # Tokenize the hypothesis and references
    hypothesis_tokens = word_tokenize(hypothesis)
    references_tokens = [word_tokenize(ref) for ref in references]

    return meteor_score(references_tokens, hypothesis_tokens)


json_path = './NWPU_RESISC45_Captions.jsonl'
f = open(json_path)
data = [json.loads(line) for line in f.readlines()]


rouge1_precision, rouge1_recall, rouge1_fscore = 0, 0, 0
rougeL_precision, rougeL_recall, rougeL_fscore = 0, 0, 0
M_score = 0

count = 0
for idx, item in enumerate(data):

    #print(item['answer'], item['annotation'])
    response = item['answer']
    reference = [item['caption0'], item['caption1'], item['caption2'], item['caption3'], item['caption4']]
    
    rouge1, rougeL = calculate_rouge(response, reference)
    rouge1_precision = rouge1_precision + rouge1
    rougeL_precision = rougeL_precision + rougeL
    
    #print(rouge_scores)
    M_score = M_score + calculate_meteor(response, reference)
    count = count + 1
    
rouge1_precision = rouge1_precision/count
rougeL_precision = rougeL_precision/count
M_score = M_score/count

print(f"ROUGE-1: Precision: {rouge1_precision:.4f}")
print(f"ROUGE-L: Precision: {rougeL_precision:.4f}")

print(f"METEOR Score: {M_score:.4f}")

