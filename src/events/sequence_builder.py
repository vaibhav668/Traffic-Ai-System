import cv2
import mediapipe as mp
import numpy as np

mp_pose=mp.solutions.pose
pose=mp_pose.Pose()

sequence=[]
SEQUENCE_LENGTH=30
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=pose.process(rgb)

    keypoints = []

    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x,lm.y,lm.z])

    else:
        keypoinnts =[0] *99

    sequence.append(keypoints)

    if len(sequence)>SEQUENCE_LENGTH:
        sequence.pop(0)


    cv2.putText(frame, f"Seq Length: {len(sequence)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sequence Builder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()

