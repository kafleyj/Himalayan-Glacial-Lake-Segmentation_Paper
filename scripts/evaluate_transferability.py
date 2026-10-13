import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from PIL import Image

import torch
import torch.nn.functional as F

from src.models import build_model


def calculate_metrics(tp, tn, fp, fn):
    eps = 1e-8
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "specificity": float(tn / (tn + fp + eps)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
    }


def predict(model, rgb, model_size, device):
    # rgb: H x W x 3, uint8
    x = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    ).float().unsqueeze(0).to(device) / 255.0

    # Same ImageNet normalization as training.
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)

    x = F.interpolate(
        x, size=(model_size, model_size),
        mode="bilinear", align_corners=False
    )
    x = (x - mean) / std

    with torch.inference_mode():
        logits = model(x)
        prob = torch.sigmoid(logits)
        prob = F.interpolate(
            prob,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False
        )

    return prob[0, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--region", required=True,
                    choices=["AS", "GL", "NA", "SA"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--halo", type=int, default=32)
    ap.add_argument("--save-samples", type=int, default=6)
    args = ap.parse_args()

    if args.tile <= 0 or args.halo < 0:
        raise ValueError("Invalid tile or halo")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu",
        weights_only=False
    )
    architecture = checkpoint["architecture"]
    model_size = int(checkpoint["size"])

    model = build_model(architecture)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(device).eval()

    root = Path(args.data_root)
    region = args.region
    image_path = root / "images" / f"{region}.tif"
    mask_path = root / "labels" / f"{region}_gt.tif"

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "samples"
    sample_dir.mkdir(exist_ok=True)

    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tile_rows = []
    sample_count = 0

    with rasterio.open(image_path) as image_ds, \
         rasterio.open(mask_path) as mask_ds:

        if image_ds.count != 3:
            raise ValueError("Expected 3 RGB bands")

        # Ground truth is aligned to the RGB image grid.
        # WarpedVRT applies nearest-neighbor reprojection
        # when AS or GL transforms differ.
        with WarpedVRT(
            mask_ds,
            crs=image_ds.crs,
            transform=image_ds.transform,
            width=image_ds.width,
            height=image_ds.height,
            resampling=Resampling.nearest,
            src_nodata=100,
            nodata=100
        ) as aligned_gt:

            h, w = image_ds.height, image_ds.width
            print(
                f"Region={region} image={w}x{h} "
                f"architecture={architecture} "
                f"model_size={model_size} device={device}",
                flush=True
            )

            for y in range(0, h, args.tile):
                for x in range(0, w, args.tile):
                    core_h = min(args.tile, h - y)
                    core_w = min(args.tile, w - x)

                    # Halo provides context around the evaluated core.
                    x0 = max(0, x - args.halo)
                    y0 = max(0, y - args.halo)
                    x1 = min(w, x + core_w + args.halo)
                    y1 = min(h, y + core_h + args.halo)

                    input_window = Window(
                        x0, y0, x1 - x0, y1 - y0
                    )
                    rgb = image_ds.read(
                        window=input_window
                    ).transpose(1, 2, 0)

                    probability = predict(
                        model, rgb, model_size, device
                    )

                    # Crop away halo: core tiles never overlap.
                    sy = y - y0
                    sx = x - x0
                    core_prob = probability[
                        sy:sy + core_h,
                        sx:sx + core_w
                    ]
                    pred = core_prob >= 0.5

                    core_window = Window(
                        x, y, core_w, core_h
                    )
                    gt = aligned_gt.read(
                        1, window=core_window
                    )

                    valid = (gt == 0) | (gt == 255)
                    positive = gt == 255

                    tp = int(np.count_nonzero(
                        pred & positive & valid
                    ))
                    tn = int(np.count_nonzero(
                        (~pred) & (~positive) & valid
                    ))
                    fp = int(np.count_nonzero(
                        pred & (~positive) & valid
                    ))
                    fn = int(np.count_nonzero(
                        (~pred) & positive & valid
                    ))

                    for key, value in zip(
                        ("tp", "tn", "fp", "fn"),
                        (tp, tn, fp, fn)
                    ):
                        totals[key] += value

                    tile_rows.append({
                        "region": region,
                        "x": x,
                        "y": y,
                        "valid_pixels": int(valid.sum()),
                        "lake_pixels": int(
                            np.count_nonzero(positive & valid)
                        ),
                        **calculate_metrics(tp, tn, fp, fn)
                    })

                    # Save a small number of lake-containing examples.
                    if sample_count < args.save_samples and \
                            np.any(positive & valid):
                        core_rgb = rgb[
                            sy:sy + core_h,
                            sx:sx + core_w
                        ]
                        prefix = (
                            f"{region}_{sample_count:02d}"
                            f"_x{x}_y{y}"
                        )
                        Image.fromarray(core_rgb).save(
                            sample_dir / f"{prefix}_rgb.png"
                        )
                        Image.fromarray(
                            np.where(positive & valid, 255, 0)
                            .astype(np.uint8)
                        ).save(
                            sample_dir / f"{prefix}_gt.png"
                        )
                        Image.fromarray(
                            (pred.astype(np.uint8) * 255)
                        ).save(
                            sample_dir / f"{prefix}_pred.png"
                        )
                        sample_count += 1

                print(
                    f"Processed rows {y}–{min(y + args.tile, h)}"
                    f" / {h}",
                    flush=True
                )

    metrics = {
        "region": region,
        "architecture": architecture,
        "checkpoint": args.checkpoint,
        "model_size": model_size,
        "tile": args.tile,
        "halo": args.halo,
        "tiles_evaluated": len(tile_rows),
        "valid_pixels": sum(
            r["valid_pixels"] for r in tile_rows
        ),
        **calculate_metrics(**totals)
    }

    with open(output / "tile_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=tile_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(tile_rows)

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("FINAL RESULTS", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
