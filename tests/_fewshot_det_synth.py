"""Synthetic scenes for the few-shot detection tests: textured parts (rotated, scaled) among look-alike distractors on
a noisy, unevenly lit background. Each scene comes with its ground truth boxes (x, y, w, h, class)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _part(kind: str, rng) -> tuple[np.ndarray, np.ndarray]:
    """An upright object picture and its mask."""
    if kind == "braket":                      # orange L-bracket with two dark holes and a brushed texture
        w, h = 92, 64
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array([[0, 0], [w - 1, 0], [w - 1, 24], [30, 24], [30, h - 1], [0, h - 1]])], 255)
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (40, 140, 225)
        yy, xx = np.mgrid[0:h, 0:w]
        img = (img + ((np.sin(xx * 0.9) > 0) * 25)[..., None]).clip(0, 255).astype(np.uint8)
        cv2.circle(img, (15, 46), 7, (30, 30, 30), -1)
        cv2.circle(img, (72, 12), 7, (30, 30, 30), -1)
        cv2.line(img, (4, 4), (86, 4), (20, 80, 150), 2)
    elif kind == "etiket":                    # blue tag: rounded plate, white stripes, a notch
        w, h = 74, 48
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (0, 0), (w - 1, h - 1), 255, -1)
        cv2.circle(m, (w - 1, h // 2), 9, 0, -1)
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (200, 110, 40)
        for x in range(8, 50, 10):
            cv2.rectangle(img, (x, 8), (x + 4, h - 9), (245, 245, 245), -1)
        cv2.circle(img, (60, 12), 5, (0, 220, 255), -1)
    else:
        raise ValueError(kind)
    return img, m


def _distractor(rng, size) -> tuple[np.ndarray, np.ndarray]:
    """Plain look-alikes: same colours, other shapes."""
    s = int(size)
    img = np.zeros((s, s, 3), np.uint8)
    m = np.zeros((s, s), np.uint8)
    col = [(40, 140, 225), (200, 110, 40), (90, 90, 90), (60, 180, 60)][int(rng.integers(4))]
    img[:] = col
    k = int(rng.integers(4))
    if k == 0:
        cv2.circle(m, (s // 2, s // 2), s // 2 - 1, 255, -1)
    elif k == 1:
        cv2.fillPoly(m, [np.array([[s // 2, 0], [s - 1, s - 1], [0, s - 1]])], 255)
    elif k == 2:
        cv2.rectangle(m, (0, s // 4), (s - 1, 3 * s // 4), 255, -1)
    else:
        cv2.ellipse(m, (s // 2, s // 2), (s // 2 - 1, s // 3), 0, 0, 360, 255, -1)
    if rng.random() < 0.5:
        for x in range(4, s - 4, 9):
            cv2.line(img, (x, 0), (x, s - 1), (230, 230, 230), 2)
    return img, m


def _paste(scene, gt_mask, img, m, cx, cy, angle, scale):
    h, w = m.shape
    diag = int(math.ceil(math.hypot(w, h) * scale)) + 4
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    M[:, 2] += (diag - 1) / 2 - np.array([(w - 1) / 2, (h - 1) / 2])
    wi = cv2.warpAffine(img, M, (diag, diag), flags=cv2.INTER_LINEAR)
    wm = cv2.warpAffine(m, M, (diag, diag), flags=cv2.INTER_LINEAR)
    x0, y0 = int(round(cx - (diag - 1) / 2)), int(round(cy - (diag - 1) / 2))
    H, W = scene.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + diag > W or y0 + diag > H:
        return None
    if gt_mask is not None and gt_mask[y0:y0 + diag, x0:x0 + diag].any():
        return None
    a = (wm.astype(np.float32) / 255.0)[..., None]
    roi = scene[y0:y0 + diag, x0:x0 + diag].astype(np.float32)
    scene[y0:y0 + diag, x0:x0 + diag] = (roi * (1 - a) + wi.astype(np.float32) * a).astype(np.uint8)
    if gt_mask is not None:
        gt_mask[y0:y0 + diag, x0:x0 + diag] |= cv2.dilate((wm > 127).astype(np.uint8), np.ones((9, 9), np.uint8))
    x, y, bw, bh = cv2.boundingRect((wm > 127).astype(np.uint8))
    return x0 + x, y0 + y, bw, bh


def scene(seed: int, kinds=("braket",), n=(2, 4), distractors=(3, 6), h=480, w=640, angle=30.0, scale=(0.8, 1.2)):
    """(BGR image, [(x, y, w, h, class)])."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 110 + 30 * np.sin(xx / 170.0 + seed) * np.cos(yy / 210.0)
    img = (base[..., None] + rng.normal(0, 9, (h, w, 3))).clip(0, 255).astype(np.uint8)
    for _ in range(6):                        # background clutter: thin lines
        p1 = tuple(int(v) for v in rng.integers(0, [w, h]))
        p2 = tuple(int(v) for v in rng.integers(0, [w, h]))
        cv2.line(img, p1, p2, tuple(int(v) for v in rng.integers(60, 180, 3)), 1)
    occ = np.zeros((h, w), np.uint8)
    boxes = []
    want = int(rng.integers(n[0], n[1] + 1))
    tries = 0
    while len(boxes) < want and tries < 300:
        tries += 1
        k = int(rng.integers(len(kinds)))
        pimg, pm = _part(kinds[k], rng)
        b = _paste(img, occ, pimg, pm, rng.uniform(60, w - 60), rng.uniform(60, h - 60),
                   rng.uniform(-angle, angle), rng.uniform(*scale))
        if b:
            boxes.append((*b, k))
    nd = int(rng.integers(distractors[0], distractors[1] + 1))
    placed, tries = 0, 0
    while placed < nd and tries < 300:
        tries += 1
        dimg, dm = _distractor(rng, rng.uniform(40, 80))
        if _paste(img, occ, dimg, dm, rng.uniform(50, w - 50), rng.uniform(50, h - 50), rng.uniform(0, 360),
                  1.0) is not None:
            placed += 1
    return img, boxes


def evaluate(dets, gt, iou_thr=0.5):
    """(true positives, false positives, false negatives) with greedy matching per class."""
    from visionflow.fewshot_det import box_iou
    used = set()
    tp = fp = 0
    for d in sorted(dets, key=lambda d: -d["score"]):
        best, bi = 0.0, -1
        for i, g in enumerate(gt):
            if i in used or g[4] != d["class_id"]:
                continue
            v = box_iou(d["box"], g[:4])
            if v > best:
                best, bi = v, i
        if best >= iou_thr:
            used.add(bi)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(used)
