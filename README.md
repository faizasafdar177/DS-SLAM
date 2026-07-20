# DS-SLAM: Dynamic-Object Masking + SLAM Pipeline

DS-SLAM combines dynamic-object masking (DINO + SAM) with ORB-SLAM2 tracking, evaluated on EuRoC, KITTI, and TUM RGB-D. The name reflects the two-stage pipeline: DINO+SAM masking, followed by SLAM.

## Pipeline

**Stage 1 — Masking (dynamic_masking/):** removes dynamic objects from each frame. See dynamic_masking/README.md.

**Stage 2 — SLAM + Evaluation:** ORB-SLAM2 runs on the masked frames, and ATE/RPE are computed against ground truth.

## Requirements

Requires a working build of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), and the Python packages in dynamic_masking/requirements.txt.

## Citation

Citation details to be added upon publication.

## License

See LICENSE file.
