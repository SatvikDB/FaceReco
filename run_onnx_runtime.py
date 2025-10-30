#!/usr/bin/env python3
"""
Run Yunet ONNX with onnxruntime and postprocess on CPU — no onnxsim required.
This script loads the ONNX model in `models/build/face_detection_yunet.onnx` and runs
inference using onnxruntime, then applies the same decoding and NMS as the DepthAI
postprocessing (host-side). Works with webcams (Camo virtual webcam) without OAK.

Usage:
  .\.venv\Scripts\python.exe run_onnx_runtime.py --input 0
  .\.venv\Scripts\python.exe run_onnx_runtime.py --model models/build/face_detection_yunet.onnx --input 0

Notes:
- Default model input resolution is inferred from the repo usage: 180x320 (HxW).
"""
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='models/build/face_detection_yunet.onnx')
parser.add_argument('--input', '-i', default='0')
parser.add_argument('--score_thresh', type=float, default=0.6)
parser.add_argument('--nms_thresh', type=float, default=0.3)
parser.add_argument('--top_k', type=int, default=50)
args = parser.parse_args()

if not os.path.isfile(args.model):
    print('Model not found:', args.model)
    exit(1)

# Model input dimensions (HxW)
NN_H = 180
NN_W = 320

# Priors generation (copied logic from YuNet)
min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
steps = [8, 16, 32, 64]
variance = [0.1, 0.2]

def prior_gen(w=NN_W, h=NN_H):
    feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
    feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
    feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
    feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
    feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]
    feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]
    priors = []
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes_k:
                    s_kx = min_size / w
                    s_ky = min_size / h
                    cx = (j + 0.5) * steps[k] / w
                    cy = (i + 0.5) * steps[k] / h
                    priors.append([cx, cy, s_kx, s_ky])
    return np.array(priors, dtype=np.float32)

PRIORS = prior_gen()

# Helpers for decoding
def decode(loc, priors, padded_size):
    # loc: Nx14
    bboxes = np.hstack((
        (priors[:, 0:2] + loc[:, 0:2] * variance[0] * priors[:, 2:4]) * padded_size,
        (priors[:, 2:4] * np.exp(loc[:, 2:4] * variance)) * padded_size
    ))
    bboxes[:, 0:2] -= bboxes[:, 2:4] / 2
    landmarks = np.hstack((
        (priors[:, 0:2] + loc[:,  4: 6] * variance[0] * priors[:, 2:4]) * padded_size,
        (priors[:, 0:2] + loc[:,  6: 8] * variance[0] * priors[:, 2:4]) * padded_size,
        (priors[:, 0:2] + loc[:,  8:10] * variance[0] * priors[:, 2:4]) * padded_size,
        (priors[:, 0:2] + loc[:, 10:12] * variance[0] * priors[:, 2:4]) * padded_size,
        (priors[:, 0:2] + loc[:, 12:14] * variance[0] * priors[:, 2:4]) * padded_size
    ))
    return np.hstack((bboxes, landmarks))

# Create ONNX runtime session
sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
print('ONNX model loaded. Provider:', sess.get_providers())

# Open capture
input_src = args.input
if str(input_src).isdigit():
    cap = cv2.VideoCapture(int(input_src), cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(input_src)
if not cap.isOpened():
    print('Unable to open capture:', input_src)
    exit(1)

print('Starting capture - press ESC or q to quit')
while True:
    ok, frame = cap.read()
    if not ok:
        print('End of stream or read failed')
        break

    # Debug: print frame info occasionally
    # (print once per 60 frames to avoid flooding)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 60 == 0:
        print(f'Frame read: shape={getattr(frame, "shape", None)}')

    img_h, img_w = frame.shape[:2]
    # Compute padded size to keep aspect ratio (same logic as YuNet)
    iwnh_ihnw = img_w * NN_H / (img_h * NN_W)
    if iwnh_ihnw >= 1:
        padded_w = img_w
        padded_h = int(round(img_h * iwnh_ihnw))
    else:
        padded_w = int(round(img_w / iwnh_ihnw))
        padded_h = img_h
    padded_size = np.array((padded_w, padded_h)).astype(int)

    padded = cv2.copyMakeBorder(frame, 0, padded_h - img_h, 0, padded_w - img_w, cv2.BORDER_CONSTANT)
    resized = cv2.resize(padded, (NN_W, NN_H), interpolation=cv2.INTER_AREA)
    # Debug windows to help diagnose black output
    cv2.imshow('NN input (resized)', resized)
    # prepare input NCHW float32
    inp = resized.astype(np.float32)
    inp = inp.transpose(2,0,1)[np.newaxis, ...]

    # Run inference
    try:
        outputs = sess.run(['loc','conf','iou'], {'input': inp})
    except Exception as e:
        print('ONNX runtime inference error:', e)
        # show shapes of input to help debugging
        print('Input shape:', inp.shape, 'dtype:', inp.dtype)
        continue
    loc = np.array(outputs[0]).reshape(-1,14)
    conf = np.array(outputs[1]).reshape(-1,2)
    iou = np.array(outputs[2]).reshape(-1)

    # compute scores same as YuNet: sqrt(conf[:,1] * iou)
    cls_scores = conf[:,1]
    iou_scores = np.clip(iou, 0.0, 1.0)
    scores = np.sqrt(cls_scores * iou_scores)[:, np.newaxis]

    dets = decode(loc, PRIORS, padded_size)
    dets = np.hstack((dets, scores))

    # NMS using OpenCV
    bboxes_list = dets[:, 0:4].tolist()
    scores_list = dets[:, -1].tolist()
    keep_idx = cv2.dnn.NMSBoxes(bboxes_list, scores_list, args.score_thresh, args.nms_thresh, top_k=args.top_k)
    faces = []
    if len(keep_idx) > 0:
        keep = np.array(keep_idx).reshape(-1)
        faces = dets[keep]

    # Draw results (map back to original coordinates: since padding added on bottom/right only, origin is unchanged)
    out = frame.copy()
    print(f'Detections: {len(faces) if len(faces)>0 else 0}')
    for f in faces:
        x, y, w, h = f[0:4].astype(int)
        score = f[-1]
        if score < args.score_thresh:
            continue
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        for li in range(5):
            lx = int(f[4 + li*2])
            ly = int(f[4 + li*2 + 1])
            cv2.circle(out, (lx,ly), 2, (0,0,255), -1)
        cv2.putText(out, f"{score:.2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow('Yunet ONNX runtime', out)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Exited')
