import cv2
import time

VIDEO_SOURCE=0

cap=cv2.VideoCapture(VIDEO_SOURCE)

prev_time=0

while True:
    ret,frame = cap.read()

    if not ret:
        break
    curr_time=time.time()
    fps=1/(curr_time-prev_time) if prev_time !=0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Video Stream",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows