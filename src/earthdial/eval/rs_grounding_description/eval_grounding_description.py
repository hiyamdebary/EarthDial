import re
import os
import json
import numpy as np
from shapely.geometry import Polygon



def get_rotated_box_vertices(bx_left, by_top, bx_right, by_bottom, theta):
    """
    Convert a rotated bounding box to its vertices.

    Parameters:
    - bx_left, by_top, bx_right, by_bottom: coordinates of the bounding box before rotation
    - theta: rotation angle in degrees

    Returns:
    - List of vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    cx = (bx_left + bx_right) / 2
    cy = (by_top + by_bottom) / 2

    width = bx_right - bx_left
    height = by_bottom - by_top

    rect = np.array([
        [bx_left, by_top],
        [bx_right, by_top],
        [bx_right, by_bottom],
        [bx_left, by_bottom]
    ])

    rotation_matrix = np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
    ])

    # Shift to the origin (center), apply rotation, then shift back
    rect_centered = rect - np.array([cx, cy])
    rotated_rect = np.dot(rect_centered, rotation_matrix) + np.array([cx, cy])

    return rotated_rect.tolist()

def calculate_iou(box1, box2):
    """
    Calculate IoU of two rotated bounding boxes.

    Parameters:
    - box1, box2: list of vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    Returns:
    - IoU value
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def parse_bboxes1(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Find all bounding box patterns: each pattern is a list of five integers within square brackets
        bounding_boxes = re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]", text)
        
        # Convert each bounding box from string tuples to lists of integers
        bounding_boxes = [list(map(int, bbox)) for bbox in bounding_boxes]
    elif isinstance(text, list):
        # If input is already a list, assume it is in the correct format
        bounding_boxes = text
    else:
        raise ValueError("Input should be either a string or a list of bounding boxes")

    return bounding_boxes


def parse_bboxes(text):
    # Find all bounding box patterns: each pattern is a list of five integers within square brackets
    bounding_boxes = re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]", text)
    
    # Convert each bounding box from string tuples to lists of integers
    bounding_boxes = [list(map(int, bbox)) for bbox in bounding_boxes]

    return bounding_boxes



json_path = './UCAS_AOD.jsonl'



f = open(json_path)
threshold = 50
data = [json.loads(line) for line in f.readlines()]

correct = total_pred = total_gt = 0
count = 0

for idx, item in enumerate(data):    
    
    pred_bboxes = parse_bboxes(item['answer'])
    gt_bboxes = parse_bboxes(item['obj_ids'])
    
    if len(gt_bboxes) < threshold:
        count += 1        
        total_pred += len(pred_bboxes)
        for pred_box in pred_bboxes:
            pred_vertices = get_rotated_box_vertices(*pred_box)
            for gt_box in gt_bboxes:
                gt_vertices = get_rotated_box_vertices(*gt_box)
                iou = calculate_iou(pred_vertices, gt_vertices)
                if iou >= 0.5:
                    correct += 1
                    break  # Count each prediction only once
    
print(count, idx)
if correct>1:
    print(f'Precision @0.5: {correct / total_pred} \n')

correct = total_pred = total_gt = 0

for idx, item in enumerate(data):    
    
    pred_bboxes = parse_bboxes(item['answer'])
    gt_bboxes = parse_bboxes(item['obj_ids'])
        
    if len(gt_bboxes) < threshold:
        total_pred += len(pred_bboxes)
        for pred_box in pred_bboxes:
            pred_vertices = get_rotated_box_vertices(*pred_box)
            for gt_box in gt_bboxes:
                gt_vertices = get_rotated_box_vertices(*gt_box)
                iou = calculate_iou(pred_vertices, gt_vertices)
                #print(iou)
                if iou >= 0.25:
                    correct += 1
                    break  # Count each prediction only once

if correct>1:
    print(f'Precision @0.25: {correct / total_pred} \n')




####################################### Description Evaluation ########################################


import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer



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

def parse_description(input_string):
    # Remove <ref> and </ref> tags
    text_without_ref_tags = re.sub(r"<\/?ref>", "", input_string)
    
    # Remove <box> and </box> tags and the bounding box contents inside
    parsed_string = re.sub(r"<box>\[\[.*?\]\]<\/box>|\[\[.*?\]\]", "", text_without_ref_tags)

    
    # Clean up extra spaces and commas if any
    #parsed_string = re.sub(r'\s+', ' ', parsed_string).strip()
    
    return parsed_string




rouge1_precision, rouge1_recall, rouge1_fscore = 0, 0, 0
rougeL_precision, rougeL_recall, rougeL_fscore = 0, 0, 0
M_score = 0
count = 0

for idx, item in enumerate(data):

    pred_bboxes = parse_bboxes(item['answer'])
    gt_bboxes = parse_bboxes(item['groundtruth'])
        
    if len(gt_bboxes) < threshold:
        response = parse_description(item['answer'])
        reference = parse_description(item['groundtruth'])

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

print(f"ROUGE-1: Precision: {rouge1_precision:.4f}, Recall: {rouge1_recall:.4f}, F1 Score: {rouge1_fscore:.4f}")
print(f"ROUGE-L: Precision: {rougeL_precision:.4f}, Recall: {rougeL_recall:.4f}, F1 Score: {rougeL_fscore:.4f}")

M_score = M_score/count
print(f"METEOR Score: {M_score:.4f}")

