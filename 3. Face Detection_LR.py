# Face Detection_LR.py

import cv2
import mediapipe as mp
import drawing_utils_adh as mp_drawing
#mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# 웹캠용 코드
cap = cv2.VideoCapture(0)  # 괄호 안의 숫자는 웹캠 번호이다. 
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break
    image = cv2.resize(image, (800, 600))
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # 얼굴에 사각형, 눈, 코, 귀, 입에 빨간점을 표시
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        rsp, rep, pos=mp_drawing.draw_detection(image, detection)   
        if rsp==None or rep==None: break

    lr_text=""
        # 0: 우측 눈 / 1: 좌측 눈 / 2: 코 / 3: 입 / 4: 우측 귀 / 5: 좌측 귀
    if pos[0] and pos[1] and pos[2] and pos[3] and pos[4] and pos[5]:
            #좌-우
        r_dis=(pos[0][0]-pos[4][0])**2 + (pos[0][1]-pos[4][1])**2
        l_dis=(pos[1][0]-pos[5][0])**2 + (pos[1][1]-pos[5][1])**2

        lr_threshold=1000
            
        if r_dis-l_dis>lr_threshold: lr_text="left"
        elif -lr_threshold<=r_dis-l_dis<=lr_threshold: lr_text="front"
        else: lr_text="right"

    image=cv2.flip(image, 1)
    font=cv2.FONT_HERSHEY_SIMPLEX
    if lr_text: cv2.putText(image,lr_text,(350,80),font,2,(127,0,255),5,cv2.LINE_AA)    
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
