#!/usr/bin/env python3
"""
Capture one frame, run ONNX model with onnxruntime, and print debugging info.
"""
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import sys

MODEL = 'models/build/face_detection_yunet.onnx'
if not __import__('os').path.isfile(MODEL):
    print('Model not found:', MODEL)
    sys.exit(1)

# Try multiple backends and warm up the camera (some virtual cams need several frames)
def try_open(index=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for b in backends:
        cap = cv2.VideoCapture(index, b)
        if not cap.isOpened():
            cap.release()
            continue
        # warm up
        frame = None
        ok = False
        for _ in range(30):
            ok, f = cap.read()
            if ok and f is not None and f.any():
                frame = f
                break
        if frame is None:
            # try to take last frame anyway
            ok, frame = ok, f if 'f' in locals() else (False, None)
        if ok:
            print(f'Opened camera with backend {b}')
            return cap, ok, frame
        cap.release()
    return None, False, None

cap, ok, frame = try_open(0)
if cap is None or not ok or frame is None:
    print('Failed to open camera or no non-empty frames received (try closing other apps using the camera)')
    sys.exit(1)

# close the capture now (we only needed one frame for debug)
cap.release()

print('Frame shape:', frame.shape, 'dtype:', frame.dtype, 'min/max:', frame.min(), frame.max())
# Resize/pad to 180x320 as in runtime script
NN_H, NN_W = 180, 320
img_h, img_w = frame.shape[:2]
iwnh_ihnw = img_w * NN_H / (img_h * NN_W)
if iwnh_ihnw >= 1:
    padded_w = img_w
    padded_h = int(round(img_h * iwnh_ihnw))
else:
    padded_w = int(round(img_w / iwnh_ihnw))
    padded_h = img_h
padded = cv2.copyMakeBorder(frame, 0, padded_h - img_h, 0, padded_w - img_w, cv2.BORDER_CONSTANT)
resized = cv2.resize(padded, (NN_W, NN_H), interpolation=cv2.INTER_AREA)
print('Resized shape:', resized.shape, 'min/max:', resized.min(), resized.max(), 'mean:', resized.mean())

# Prepare input
inp = resized.astype(np.float32).transpose(2,0,1)[np.newaxis, ...]
print('Input tensor shape (NCHW):', inp.shape, 'dtype:', inp.dtype, 'min/max:', inp.min(), inp.max(), 'mean:', inp.mean())

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('Loaded ONNX providers:', sess.get_providers())

input_name = sess.get_inputs()[0].name
print('Model input name:', input_name)

out_names = [o.name for o in sess.get_outputs()]
print('Model outputs:', out_names)

outs = sess.run(out_names, {input_name: inp})
for name, o in zip(out_names, outs):
    na = np.array(o)
    print(f"Output '{name}' shape={na.shape} dtype={na.dtype} min={na.min()} max={na.max()} mean={na.mean()}")

# Quick attention: show first 5 loc rows, conf rows, iou
loc = np.array(outs[0]).reshape(-1,14)
conf = np.array(outs[1]).reshape(-1,2)
iou = np.array(outs[2]).reshape(-1)
print('loc[0]:', loc[0])
print('conf[0]:', conf[0])
print('iou[0:10]:', iou[:10])

print('Done')
