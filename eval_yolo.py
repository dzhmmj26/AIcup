# === 使用者設定 ===
DATASET_DIR    = 'datasets/test/images'          # 影像資料夾
GT_LABEL_DIR   = 'datasets/test/labels'          # 標準答案 label 資料夾
PRED_LABEL_DIR = 'runs/detect/val2/labels'         # 預測結果 label 資料夾 (注意 run / runs)
YAML_PATH      = 'aortic_valve_colab.yaml'       # YOLO 訓練時用的 data.yaml
OUTPUT_DIR     = 'analysis_output/visualizations'  # 視覺化輸出路徑 (只放圖)
REPORT_PATH    = 'analysis_output/eval_report.md'  # Markdown 報告輸出路徑
IOU_THRESHOLD  = 0.5                              # IoU 門檻

# --------------------------------------------------------------------
# 下面開始是程式本體，不用改
# --------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2

try:
    import yaml
except ImportError:
    yaml = None


def load_class_names(data_yaml: Optional[Path]) -> Dict[int, str]:
    if data_yaml is None or not data_yaml.exists() or yaml is None:
        return {}
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    else:
        return {}


def read_yolo_labels(path: Path, is_pred: bool) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 6 if is_pred else 5), dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    boxes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        if is_pred:
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            boxes.append([cls, x, y, w, h, conf])
        else:
            boxes.append([cls, x, y, w, h])
    if not boxes:
        return np.zeros((0, 6 if is_pred else 5), dtype=float)
    return np.array(boxes, dtype=float)


def box_iou_yolo_norm(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    b1: [N, 4] (x, y, w, h) normalized
    b2: [M, 4]
    return: [N, M] IoU
    """
    if b1.size == 0 or b2.size == 0:
        return np.zeros((b1.shape[0], b2.shape[0]), dtype=float)

    # convert to xyxy
    def to_xyxy(b):
        x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = to_xyxy(b1)
    x1_2, y1_2, x2_2, y2_2 = to_xyxy(b2)

    N = b1.shape[0]
    M = b2.shape[0]
    ious = np.zeros((N, M), dtype=float)
    for i in range(N):
        xx1 = np.maximum(x1_1[i], x1_2)
        yy1 = np.maximum(y1_1[i], y1_2)
        xx2 = np.minimum(x2_1[i], x2_2)
        yy2 = np.minimum(y2_1[i], y2_2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        area1 = (x2_1[i] - x1_1[i]) * (y2_1[i] - y1_1[i])
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        iou = np.where(union > 0, inter / union, 0.0)
        ious[i] = iou
    return ious


def find_image(img_dir: Path, stem: str) -> Optional[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    class_names: Dict[int, str],
    color,
    thickness: int = 1,
    confidences: Optional[np.ndarray] = None,
    prefix: str = ""
) -> np.ndarray:
    h, w = img.shape[:2]
    for i in range(boxes.shape[0]):
        x, y, bw, bh = boxes[i]
        cls_id = int(classes[i])
        x1 = int((x - bw / 2.0) * w)
        y1 = int((y - bh / 2.0) * h)  # 注意這裡要用 bh
        x2 = int((x + bw / 2.0) * w)
        y2 = int((y + bh / 2.0) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = class_names.get(cls_id, str(cls_id))
        if prefix:
            label = f"{prefix}{label}"
        if confidences is not None and len(confidences) == boxes.shape[0]:
            label = f"{label} {confidences[i]:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return img


def main():
    # 處理路徑物件
    img_dir = Path(DATASET_DIR)
    gt_dir = Path(GT_LABEL_DIR)
    pred_dir = Path(PRED_LABEL_DIR)
    data_yaml = Path(YAML_PATH)
    vis_dir = Path(OUTPUT_DIR)
    report_path = Path(REPORT_PATH)

    # 建立輸出資料夾
    vis_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(data_yaml)

    gt_files = sorted([p for p in gt_dir.glob("*.txt")])
    pred_files = sorted([p for p in pred_dir.glob("*.txt")])

    gt_stems = {p.stem for p in gt_files}
    pred_stems = {p.stem for p in pred_files}
    all_stems = sorted(gt_stems.union(pred_stems))

    total_images = len(all_stems)
    print(f"Found {total_images} images (union of GT and predictions).")

    # global counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_dups = 0
    total_gt = 0
    total_pred = 0
    iou_sum = 0.0
    iou_count = 0

    # per-class counters: cls -> [tp, fp, fn]
    per_class = {}

    for stem in all_stems:
        gt_path = gt_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"

        gt = read_yolo_labels(gt_path, is_pred=False)   # [N_gt, 5]
        pred = read_yolo_labels(pred_path, is_pred=True)  # [N_pred, 6]

        gt_cls = gt[:, 0].astype(int) if gt.size else np.array([], dtype=int)
        gt_boxes = gt[:, 1:5] if gt.size else np.zeros((0, 4), dtype=float)

        pred_cls = pred[:, 0].astype(int) if pred.size else np.array([], dtype=int)
        pred_boxes = pred[:, 1:5] if pred.size else np.zeros((0, 4), dtype=float)
        pred_conf = pred[:, 5] if pred.size and pred.shape[1] >= 6 else np.array([], dtype=float)

        total_gt += gt_boxes.shape[0]
        total_pred += pred_boxes.shape[0]

        if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
            continue

        # sort predictions by confidence descending (for matching)
        if pred_boxes.shape[0] > 0 and pred_conf.size == pred_boxes.shape[0]:
            order = np.argsort(-pred_conf)
            pred_boxes = pred_boxes[order]
            pred_cls = pred_cls[order]
            pred_conf = pred_conf[order]

        matched_gt = set()
        matched_pred = set()
        dup_pred_indices = set()

        if gt_boxes.shape[0] > 0 and pred_boxes.shape[0] > 0:
            ious = box_iou_yolo_norm(pred_boxes, gt_boxes)
        else:
            ious = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=float)

        # match
        for pi in range(pred_boxes.shape[0]):
            cls_p = pred_cls[pi]
            best_iou = 0.0
            best_gi = -1
            for gi in range(gt_boxes.shape[0]):
                if gt_cls[gi] != cls_p:
                    continue
                iou = ious[pi, gi]
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= IOU_THRESHOLD:
                if best_gi not in matched_gt:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                    total_tp += 1
                    iou_sum += best_iou
                    iou_count += 1
                    pc = per_class.setdefault(cls_p, [0, 0, 0])
                    pc[0] += 1  # tp
                else:
                    # duplicate prediction for an already matched gt
                    total_fp += 1
                    total_dups += 1
                    dup_pred_indices.add(pi)
                    pc = per_class.setdefault(cls_p, [0, 0, 0])
                    pc[1] += 1  # fp
            else:
                # no match: false positive
                total_fp += 1
                pc = per_class.setdefault(cls_p, [0, 0, 0])
                pc[1] += 1  # fp

        # false negatives: gt not matched
        for gi in range(gt_boxes.shape[0]):
            if gi not in matched_gt:
                total_fn += 1
                cls_g = gt_cls[gi]
                pc = per_class.setdefault(cls_g, [0, 0, 0])
                pc[2] += 1  # fn

        # visualization: 每張都畫
        img_path = find_image(img_dir, stem)
        if img_path is not None and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                # draw gt (green, 細框)
                if gt_boxes.shape[0] > 0:
                    img = draw_boxes(
                        img,
                        gt_boxes,
                        gt_cls,
                        class_names,
                        color=(0, 255, 0),
                        thickness=1,
                        confidences=None,
                        prefix="GT:"
                    )
                # draw predictions (red, 細框)
                if pred_boxes.shape[0] > 0:
                    img = draw_boxes(
                        img,
                        pred_boxes,
                        pred_cls,
                        class_names,
                        color=(0, 0, 255),
                        thickness=1,
                        confidences=pred_conf if pred_conf.size == pred_boxes.shape[0] else None,
                        prefix="P:"
                    )
                out_img_path = vis_dir / f"{stem}.jpg"
                cv2.imwrite(str(out_img_path), img)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    # Markdown report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# YOLO 模型錯誤分析報告\n\n")
        f.write("## 基本資訊\n\n")
        f.write(f"- IoU 門檻：`{IOU_THRESHOLD}`\n")
        f.write(f"- 影像數量（GT 與預測檔名聯集）：`{total_images}`\n")
        f.write(f"- 總標註物件數 (GT)：`{total_gt}`\n")
        f.write(f"- 總預測框數：`{total_pred}`\n\n")

        f.write("## 整體表現\n\n")
        f.write(f"- True Positive (TP)：`{total_tp}`\n")
        f.write(f"- False Positive (FP)：`{total_fp}`\n")
        f.write(f"- False Negative (FN)：`{total_fn}`\n")
        f.write(f"- 同一物件被多次預測（重複框數量）：`{total_dups}`\n")
        f.write(f"- Precision：`{precision:.4f}`\n")
        f.write(f"- Recall：`{recall:.4f}`\n")
        f.write(f"- F1-score：`{f1:.4f}`\n")
        f.write(f"- 匹配框平均 IoU：`{mean_iou:.4f}`\n\n")

        f.write("### 指標對應題目要求\n\n")
        f.write("1. **預測與實際的重疊程度**：使用匹配到的框之平均 IoU 來衡量（上面的 *平均 IoU*）。\n")
        f.write("2. **同一張圖片是否出現多個框指向同一個物件**：統計為「同一物件被多次預測（重複框數量）」；同一 GT 被多個預測框 IoU ≥ 門檻時，除了第一個 TP，其餘皆視為重複 FP。\n")
        f.write("3. **實際存在但沒有被預測到**：統計為 FN（False Negative）。\n")
        f.write("4. **被預測到但實際並不存在**：統計為 FP（False Positive，包括重複框）。\n\n")

        f.write("## 各類別指標\n\n")
        f.write("| 類別ID | 類別名稱 | TP | FP | FN | Precision | Recall |\n")
        f.write("|--------|----------|----|----|----|-----------|--------|\n")
        for cls_id, (tp_c, fp_c, fn_c) in sorted(per_class.items(), key=lambda x: x[0]):
            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            name = class_names.get(cls_id, str(cls_id))
            f.write(
                f"| {cls_id} | {name} | {tp_c} | {fp_c} | {fn_c} | {prec_c:.4f} | {rec_c:.4f} |\n"
            )

        f.write("\n## 視覺化結果說明\n\n")
        f.write(f"- 已在 `{vis_dir}` 資料夾輸出疊加 *標準答案框 (綠色)* 與 *預測框 (紅色)* 的影像，框線設定為細線方便觀察差異。\n")
        f.write("- 圖中的文字標籤：`GT:類別` 表示標註框，`P:類別 分數` 表示模型預測框與其信心分數。\n")
        f.write("- 可透過逐張檢視視覺化影像，搭配上面的 TP/FP/FN 統計，進一步分析誤判情況。\n")

    print(f"Analysis finished. Markdown report saved to: {report_path}")
    print(f"Visualization images saved to: {vis_dir}")


if __name__ == "__main__":
    main()
