import cv2
for i in range(7):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    print(f"Index {i}: opened={cap.isOpened()}, read_ok={ok}")
    cap.release()
