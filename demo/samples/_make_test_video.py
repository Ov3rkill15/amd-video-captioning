import cv2
import numpy as np

path = "demo/samples/test_input.mp4"
w, h, fps, seconds = 320, 240, 10, 4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

colors = [(60, 60, 220), (60, 220, 60), (220, 60, 60), (220, 220, 60)]
for i in range(fps * seconds):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    color = colors[(i // fps) % len(colors)]
    frame[:] = color
    cx = int(w * (i / (fps * seconds)))
    cv2.circle(frame, (cx, h // 2), 30, (255, 255, 255), -1)
    cv2.putText(frame, f"t={i/fps:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print(f"wrote {path}")
