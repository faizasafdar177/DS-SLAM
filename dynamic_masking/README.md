# Dynamic-Object Masking Module

This module removes dynamic objects (e.g., pedestrians, vehicles) from monocular, stereo, and RGB-D image sequences before feature-based visual SLAM tracking. It is Stage 1 of the DS-SLAM pipeline.

## Modes

| Mode | Description |
|---|---|
| dino_only | Detects dynamic objects with Grounding DINO and blacks out the detected rectangles. |
| sam_only | Segments the whole image automatically with SAM; blacks out every region above a minimum size. |
| dino_sam | Grounding DINO finds the object; SAM then traces its precise outline, and only that shape is blacked out. |

## Usage

```bash
python3 src/mask_generator.py --dataset euroc --sequence MH_01_easy --mode dino_sam
```

Supported --dataset: euroc, kitti, tum
Supported --mode: dino_only, sam_only, dino_sam

Output is written to masked_data/<dataset>/<sequence>/<mode>/<camera>/, keeping original filenames.
