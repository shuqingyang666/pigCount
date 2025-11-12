"""
map_reward

date : 2025/11/10
"""
import torch

def box_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else torch.tensor(0.0)

def ap50_95(pred_boxes, gt_boxes):
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0, 0

    ious = []
    for pb in pred_boxes:
        best = 0
        for gb in gt_boxes:
            best = max(best, box_iou(pb, gb).item())
        ious.append(best)

    ious = torch.tensor(ious)

    ap50 = (ious > 0.5).float().mean().item()
    ap95 = (ious > 0.95).float().mean().item()
    map_50_95 = (ap50 + ap95) / 2
    return ap50, map_50_95
