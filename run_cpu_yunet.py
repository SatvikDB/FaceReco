#!/usr/bin/env python3
"""
Simple CPU Yunet runner using OpenCV's FaceDetectorYN (ONNX)
This allows running the face detector on a normal webcam (eg. Phone Link / Your Phone)
without requiring a DepthAI/OAK device.

Usage examples (PowerShell):
  .\.venv\Scripts\python.exe run_cpu_yunet.py           # try webcam 0
  .\.venv\Scripts\python.exe run_cpu_yunet.py --input 1 # webcam index 1
  .\.venv\Scripts\python.exe run_cpu_yunet.py --input "rtsp://..." # RTSP stream

Notes:
- The script looks for an ONNX Yunet model at 'models/build/face_detection_yunet.onnx' first,
  then at 'models/face_detection_yunet_180x320_sh4.onnx' (common build location).
- If OpenCV doesn't provide FaceDetectorYN in your build, it will print a helpful message.
"""
import argparse
import cv2
import os
import sys

# Candidate ONNX paths
CANDIDATES = [
    os.path.join("models", "build", "face_detection_yunet.onnx"),
    os.path.join("models", "face_detection_yunet_180x320.onnx"),
]

parser = argparse.ArgumentParser(description="Run Yunet (ONNX) on a normal webcam using OpenCV FaceDetectorYN")
parser.add_argument("--input", "-i", default="0",
                    help="Webcam index (0,1,...) or video/stream URL (default=0)")
parser.add_argument("--model", "-m", default=None, help="Path to Yunet ONNX model (optional)")
parser.add_argument("--width", type=int, default=640, help="Display/capture width (default=640)")
parser.add_argument("--height", type=int, default=360, help="Display/capture height (default=360)")
parser.add_argument("--score_thresh", type=float, default=0.6, help="Score threshold")
args = parser.parse_args()

# Resolve model
model_path = None
if args.model:
    if os.path.isfile(args.model):
        model_path = args.model
    else:
        print(f"Provided model not found: {args.model}")
        sys.exit(1)
else:
    for p in CANDIDATES:
        if os.path.isfile(p):
            model_path = p
            break

if model_path is None:
    print("Could not find Yunet ONNX model. Looked for:")
    for p in CANDIDATES:
        print("  ", p)
    print("You can generate or place an ONNX model in one of those paths, or pass --model <path>")
    sys.exit(1)

print("Using model:", model_path)

# Check FaceDetectorYN availability
if not hasattr(cv2, 'FaceDetectorYN_create'):
    print("Your OpenCV build doesn't expose FaceDetectorYN_create().")
    print("Install a recent opencv-python (>=4.5.1) that includes the extra face modules, or use the repo's demo with an OAK device.")
    sys.exit(1)

# Prepare capture
input_src = args.input
if str(input_src).isdigit():
    idx = int(input_src)
    # Prefer DirectShow backend on Windows
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(input_src)

if not cap.isOpened():
    print(f"Unable to open capture source: {input_src}")
    sys.exit(1)

# Try to set size (best-effort)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

# Create the detector using the model. The input size is the net input size (W,H)
net_input_size = (320, 180)  # the ONNX Yunet default used in this repo is 180x320 (HxW)
# FaceDetectorYN takes (model, config, input_size) where config is an empty string for ONNX
try:
    detector = cv2.FaceDetectorYN_create(model_path, "", (net_input_size[1], net_input_size[0]))
except Exception as e:
    print("Failed to create FaceDetectorYN:", e)
    print("Ensure your OpenCV build supports FaceDetectorYN and that model is compatible.")
    sys.exit(1)

print("Starting capture. Press ESC or 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("End of stream or cannot read frame")
        break

    # FaceDetectorYN expects BGR images. It returns faces (N x 15): x, y, w, h, landmarks(10), score
    faces = detector.detect(frame)
    # detect returns (faces, bboxes) depending on OpenCV versions; handle both
    if isinstance(faces, tuple) and len(faces) == 2:
        faces = faces[1]

    if faces is not None and len(faces) > 0:
        for f in faces:
            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = f[14]
            if score < args.score_thresh:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw landmarks (5 points)
            for li in range(5):
                lx = int(f[4 + li * 2])
                ly = int(f[4 + li * 2 + 1])
                cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
            cv2.putText(frame, f"{score:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Yunet (CPU)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited")
