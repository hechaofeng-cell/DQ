"""SAM box-prompt segmentation."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_json
from .schema import geometry_from_mask, pad_box


def select_foreground_mask(masks, scores, box):
    best = int(scores.argmax())
    mask = masks[best].astype(bool)
    height, width = mask.shape
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / (width * height)
    ys, xs = np.nonzero(mask)
    span_width = (xs.max() - xs.min() + 1) / width if len(xs) else 0
    span_height = (ys.max() - ys.min() + 1) / height if len(ys) else 0
    boundary = np.concatenate((mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]))
    boundary_occupancy = float(boundary.mean())
    complemented = bool(box_area > .5 and span_width >= .95 and span_height >= .95 and boundary_occupancy > .25)
    if complemented:
        rectangle = np.zeros_like(mask)
        x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
        x2, y2 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
        rectangle[y1:y2, x1:x2] = True
        candidate = rectangle & ~mask
        if candidate.any():
            mask = candidate
        else:
            complemented = False
    return best, mask, complemented


def run_sam(config, rows, local_results, output, protocol_hash, on_progress):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if not torch.cuda.is_available():
        raise ValueError("cuda_unavailable_for_sam")
    model = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"])
    model.to("cuda").eval()
    predictor = SamPredictor(model)
    parsed_dir, mask_dir = output / "parsed" / "sam", output / "masks"
    overlay_dir = output / "mask_overlays"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, row in enumerate(rows, 1):
        path = parsed_dir / f"{row['image_id']}.json"
        if path.exists():
            saved = __import__("json").loads(path.read_text())
            if saved.get("protocol_hash") != protocol_hash:
                raise ValueError("sam_resume_protocol_mismatch")
            results[row["image_id"]] = saved["result"]
            on_progress(index, len(rows), row["image_id"], None)
            continue
        if row["image_id"] not in local_results:
            on_progress(index, len(rows), row["image_id"], "missing_local_semantic")
            continue
        started = time.monotonic()
        try:
            image = np.asarray(Image.open(row["path"]).convert("RGB"))
            height, width = image.shape[:2]
            box = np.asarray(pad_box(local_results[row["image_id"]]["target_bbox_hint"], width, height))
            predictor.set_image(image)
            transformed = predictor.transform.apply_boxes_torch(
                torch.as_tensor(box[None, :], dtype=torch.float, device="cuda"), image.shape[:2]
            )
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed,
                multimask_output=True, return_logits=False,
            )
            candidate_masks = masks[0].detach().cpu().numpy().astype(bool)
            candidate_scores = scores[0].detach().cpu().numpy()
            best, mask, complemented = select_foreground_mask(candidate_masks, candidate_scores, box)
            result = geometry_from_mask(mask)
            result.update({"sam_score": float(scores[0, best].item()),
                           "background_complement_applied": complemented,
                           "elapsed_seconds": time.monotonic() - started})
            mask_image = Image.fromarray(mask.astype("uint8") * 255)
            mask_temp = mask_dir / f"{row['image_id']}.png.tmp"
            mask_image.save(mask_temp, format="PNG")
            mask_temp.replace(mask_dir / f"{row['image_id']}.png")
            overlay = image.copy()
            overlay[mask] = (overlay[mask].astype(np.uint16) * 2 // 5 + np.array([153, 0, 92])).clip(0, 255)
            overlay_image = Image.fromarray(overlay.astype("uint8"))
            overlay_temp = overlay_dir / f"{row['image_id']}.jpg.tmp"
            overlay_image.save(overlay_temp, format="JPEG", quality=95)
            overlay_temp.replace(overlay_dir / f"{row['image_id']}.jpg")
            atomic_json(path, {"protocol_hash": protocol_hash, "result": result})
            results[row["image_id"]] = result
            on_progress(index, len(rows), row["image_id"], None)
        except Exception as exc:
            on_progress(index, len(rows), row["image_id"], f"{type(exc).__name__}: {exc}")
    del predictor, model
    torch.cuda.empty_cache()
    return results
