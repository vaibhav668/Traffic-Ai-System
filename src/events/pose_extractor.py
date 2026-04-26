import cv2
import mediapipe as mp

mp_pose=mp.solutions.pose
pose=mp_pose.Pose()

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=pose.process(rgb)

    if result.pose_landmarks :
        for lm in result.pose_landmarks.landmark:
            h,w,_=frame.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)


    cv2.imshow("Pose",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows