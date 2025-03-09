import numpy as np
from shapely.geometry import Polygon
import json
import re

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
    """Extract bounding boxes from the string."""
    # Extract the list of bounding boxes using regex
    bbox_list = re.findall(r'\[\[(.*?)\]\]', bbox_str)
    
    bboxes = []
    for bbox in bbox_list:
        # Clean the bbox by stripping any spaces and unwanted characters
        bbox_clean = [x.strip() for x in bbox.split(',')]  # Strip spaces around values
        #print(bbox_clean)
        try:
            # Convert the cleaned strings to integers
            bbox_int = list(map(int, bbox_clean))
            bboxes.append(bbox_int)
        except ValueError as e:
            print(f"Error converting {bbox_clean} to integers: {e}")
            # Handle the error as needed (e.g., skip or log the invalid bbox)
    
    return bboxes


jsonl_path = './GeoChat.jsonl'

f = open(jsonl_path)
data = [json.loads(line) for line in f.readlines()]


# Initialize recall tracking
objs_types = ['small', 'medium', 'large', 'single', 'multiple']

pred_count = {obj_type: 0 for obj_type in objs_types}  # Store recall for each task
total_obj_counts = {obj_type: 0 for obj_type in objs_types}    # Count occurrences per task

for i, output in enumerate(data):
    
    pred_bboxes = parse_bboxes(output['answer'])
    gt_bboxes = parse_bboxes(output['gt_bbox'])
    obj_type = output['size_group']
    
    total_obj_counts[obj_type] += len(pred_bboxes)

    if len(gt_bboxes) == 1:
        total_obj_counts['single'] += len(pred_bboxes)
    else:
        total_obj_counts['multiple'] += len(pred_bboxes)

    for pred_box in pred_bboxes:
        
        try:
            pred_vertices = get_rotated_box_vertices(*pred_box)
            for gt_box in gt_bboxes:
                gt_vertices = get_rotated_box_vertices(*gt_box)
                iou = calculate_iou(pred_vertices, gt_vertices)   
                if iou >= 0.5:                        
                    pred_count[obj_type] += 1
                    if len(gt_bboxes) == 1:
                        pred_count['single'] += 1
                    else:
                        pred_count['multiple'] += 1
                    break  # Count each prediction only once
        except:
            pass
    

# Compute final recall per task type
for obj_type in pred_count:
    if pred_count[obj_type] > 0:
        pred_count[obj_type] /= total_obj_counts[obj_type]  # Average recall per task type

# Print results
for obj_type, precision in pred_count.items():
    print(f"{obj_type}: Precision = {precision:.4f}")
