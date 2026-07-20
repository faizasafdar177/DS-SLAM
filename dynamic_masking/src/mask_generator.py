import os
import sys
import time
import csv
import argparse
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_utils

BASE = os.path.expanduser("~/slam2_dino_sam_ablation")


def get_image_dirs(dataset, sequence):
    """Returns list of (input_dir, cam_name) tuples for a given dataset/sequence."""
    if dataset == "euroc":
        return [(os.path.join(BASE, "01_data", "euroc", sequence, "mav0", "cam0", "data"), "cam0")]
    elif dataset == "kitti":
        seq_dir = os.path.join(BASE, "01_data", "kitti", "dataset", "sequences", sequence)
        return [
            (os.path.join(seq_dir, "image_0"), "image_0"),
            (os.path.join(seq_dir, "image_1"), "image_1"),
        ]
    elif dataset == "tum":
        return [(os.path.join(BASE, "01_data", "tum", sequence, "rgb"), "rgb")]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_output_dir(dataset, sequence, mode, cam_name):
    out = os.path.join(BASE, "04_dino_sam_module", "masked_data", dataset, sequence, mode, cam_name)
    os.makedirs(out, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["euroc", "kitti", "tum"])
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mode", required=True, choices=["dino_only", "sam_only", "dino_sam"])
    args = parser.parse_args()

    prompts = mask_utils.load_prompts()
    thresholds = mask_utils.load_thresholds()
    text_prompt = prompts[args.dataset]

    dino_model = None
    sam_predictor = None
    sam_auto = None

    if args.mode in ("dino_only", "dino_sam"):
        print("Loading Grounding DINO model...")
        dino_model = mask_utils.load_dino_model()

    if args.mode in ("sam_only", "dino_sam"):
        print("Loading SAM model...")
        sam_predictor, sam_auto = mask_utils.load_sam_model()

    image_dirs = get_image_dirs(args.dataset, args.sequence)

    timing_dir = os.path.join(BASE, "06_results", "timing")
    os.makedirs(timing_dir, exist_ok=True)
    timing_csv_path = os.path.join(timing_dir, f"{args.dataset}_{args.sequence}_{args.mode}_timing.csv")

    with open(timing_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cam", "frame", "dino_ms", "sam_ms", "latency_ms"])

        for input_dir, cam_name in image_dirs:
            if not os.path.isdir(input_dir):
                print(f"WARNING: {input_dir} not found, skipping {cam_name}")
                continue

            output_dir = get_output_dir(args.dataset, args.sequence, args.mode, cam_name)
            frames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            print(f"Processing {len(frames)} frames from {cam_name} ({args.mode})...")

            for i, fname in enumerate(frames):
                in_path = os.path.join(input_dir, fname)
                out_path = os.path.join(output_dir, fname)

                if os.path.exists(out_path):
                    continue

                t0 = time.time()
                dino_ms = 0.0
                sam_ms = 0.0

                if args.mode == "dino_only":
                    td0 = time.time()
                    image_source, boxes = mask_utils.dino_detect(
                        dino_model, in_path, text_prompt,
                        thresholds["box_threshold"], thresholds["text_threshold"]
                    )
                    dino_ms = (time.time() - td0) * 1000
                    image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
                    result = mask_utils.mask_with_boxes(image_bgr, boxes)

                elif args.mode == "sam_only":
                    image_bgr = cv2.imread(in_path)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    ts0 = time.time()
                    result = mask_utils.mask_with_sam_auto(
                        image_rgb, sam_auto, thresholds["sam_min_mask_area_ratio"]
                    )
                    sam_ms = (time.time() - ts0) * 1000

                elif args.mode == "dino_sam":
                    td0 = time.time()
                    image_source, boxes = mask_utils.dino_detect(
                        dino_model, in_path, text_prompt,
                        thresholds["box_threshold"], thresholds["text_threshold"]
                    )
                    dino_ms = (time.time() - td0) * 1000
                    ts0 = time.time()
                    result = mask_utils.mask_with_sam_boxes(image_source, sam_predictor, boxes)
                    sam_ms = (time.time() - ts0) * 1000

                latency_ms = (time.time() - t0) * 1000
                cv2.imwrite(out_path, result)
                writer.writerow([cam_name, fname, f"{dino_ms:.2f}", f"{sam_ms:.2f}", f"{latency_ms:.2f}"])

                if i % 200 == 0:
                    print(f"  [{cam_name}] frame {i}/{len(frames)} - {latency_ms:.1f} ms")

    print(f"Done. Timing log: {timing_csv_path}")


if __name__ == "__main__":
    main()
