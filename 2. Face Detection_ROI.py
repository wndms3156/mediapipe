# Face Detection_ROI.py

import cv2
import mediapipe as mp
#import drawing_utils as mp_drawing
import drawing_utils_adh as mp_drawing

mp_face_detection = mp.solutions.face_detection
#mp_drawing = mp.solutions.drawing_utils

# 웹캠용 코드
cap = cv2.VideoCapture(0)  # 괄호 안의 숫자는 웹캠 번호이다.
#face_detection=mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # 얼굴에 사각형, 눈, 코, 귀, 입에 빨간점을 표시
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        rsp, rep, __=mp_drawing.draw_detection(image, detection)   
        if rsp==None or rep==None: break
        roi=image[rsp[1]:rep[1] , rsp[0]:rep[0]]
        
    # 좌우반전
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))

    #ROI 설정
    roi=cv2.resize(roi, (200,200))
    cv2.imshow('roi',cv2.flip(roi,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
