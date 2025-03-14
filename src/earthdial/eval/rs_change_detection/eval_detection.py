import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
import json
import re
import os

import argparse
from colorama import Fore, init
init(autoreset=True)  # Initialize Colorama

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



def parse_bboxes(bbox_str):
    """Extract bounding boxes from a string containing multiple bbox lists."""
    # Ensure proper list format by adding a comma between consecutive brackets
    formatted_str = bbox_str.replace("][", "], [")
    
    # Extract the list of bounding boxes using regex
    bbox_list = re.findall(r'\[\s*(\d+(?:,\s*\d+)*)\s*\]', formatted_str)
    
    bboxes = []
    for bbox in bbox_list:
        try:
            # Convert the extracted string into a list of integers
            bbox_int = list(map(int, bbox.split(',')))
            bboxes.append(bbox_int)
        except ValueError as e:
            print(f"Error converting {bbox} to integers: {e}")
    
    return bboxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='xBD_object_detection')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'src/earthdial/eval/rs_change_detection/results/{ds_name}.jsonl'
        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]

        correct = total_cnt = count = 0
        for i, output in enumerate(data):
            
            pred_bboxes = parse_bboxes(output['answer'])
            gt_bboxes = parse_bboxes(output['caption0'])
            
            
            total_cnt += len(pred_bboxes)
            
            for pred_box in pred_bboxes:
                pred_box[-1] = 0
                pred_vertices = get_rotated_box_vertices(*pred_box)
                for gt_box in gt_bboxes:
                    gt_vertices = get_rotated_box_vertices(*gt_box)
                    iou = calculate_iou(pred_vertices, gt_vertices)   
                    if iou >= 0.25:
                        correct += 1
                        break  # Count each prediction only once
                    

        print('-'*40)
        
        print(f'{Fore.RED}Precision score for {ds_name} dataset: {correct / total_cnt}')
        
        print('-'*40)         

