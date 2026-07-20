import os
import json
import numpy as np
import cv2
import torch

BASE = os.path.expanduser("~/slam2_dino_sam_ablation")
DINO_REPO = os.path.join(BASE, "04_dino_sam_module", "GroundingDINO")
CKPT_DIR = os.path.join(BASE, "04_dino_sam_module", "checkpoints")
CONFIG_DIR = os.path.join(BASE, "02_configs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts():
    with open(os.path.join(CONFIG_DIR, "dino_prompts.json")) as f:
        return json.load(f)


def load_thresholds():
    with open(os.path.join(CONFIG_DIR, "thresholds.json")) as f:
        return json.load(f)


def load_dino_model():
    import sys
    sys.path.insert(0, DINO_REPO)
    from groundingdino.util.inference import load_model
    config_path = os.path.join(DINO_REPO, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    weights_path = os.path.join(CKPT_DIR, "groundingdino_swint_ogc.pth")
    model = load_model(config_path, weights_path)
    model = model.to(DEVICE)
    return model


def load_sam_model():
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    checkpoint = os.path.join(CKPT_DIR, "sam_vit_h_4b8939.pth")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    auto_generator = SamAutomaticMaskGenerator(sam)
    return predictor, auto_generator


def dino_detect(dino_model, image_path, text_prompt, box_threshold, text_threshold):
    from groundingdino.util.inference import load_image, predict
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE,
    )
    h, w, _ = image_source.shape
    if len(boxes) == 0:
        return image_source, np.zeros((0, 4), dtype=int)
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    boxes_xyxy = boxes_xyxy.round().int().clamp(min=0).numpy()
    return image_source, boxes_xyxy


def mask_with_boxes(image_bgr, boxes_xyxy):
    """dino_only mode: rectangular blackout over detected boxes."""
    out = image_bgr.copy()
    for (x1, y1, x2, y2) in boxes_xyxy:
        out[y1:y2, x1:x2] = 0
    return out


def mask_with_sam_boxes(image_rgb, predictor, boxes_xyxy):
    """dino_sam mode: DINO boxes prompt SAM for precise segmentation, then mask exact shape."""
    if len(boxes_xyxy) == 0:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    predictor.set_image(image_rgb)
    out_rgb = image_rgb.copy()
    for box in boxes_xyxy:
        box_t = torch.tensor(box, device=DEVICE).unsqueeze(0)
        transformed = predictor.transform.apply_boxes_torch(box_t, image_rgb.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        mask = masks[0, 0].cpu().numpy().astype(bool)
        out_rgb[mask] = 0
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def mask_with_sam_auto(image_rgb, auto_generator, min_area_ratio):
    """sam_only mode: no semantic guidance, mask all sizeable SAM segments (heuristic, no DINO)."""
    h, w, _ = image_rgb.shape
    total_area = h * w
    masks = auto_generator.generate(image_rgb)
    out_rgb = image_rgb.copy()
    for m in masks:
        if m["area"] / total_area >= min_area_ratio and m["area"] / total_area < 0.5:
            out_rgb[m["segmentation"]] = 0
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
