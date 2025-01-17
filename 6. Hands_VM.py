# Hands_VM.py

import cv2
import mediapipe as mp
from math import sqrt
import pynput

mouse_drag = pynput.mouse.Controller()
mouse_button = pynput.mouse.Button
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# 웹캠용
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    #print(results.multi_hand_landmarks)

    # 손가락 모양대로 그리기
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        f1_x=int(hand_landmarks.landmark[4].x*100)
        f1_y=int(hand_landmarks.landmark[4].y*100)
        f2_x=int(hand_landmarks.landmark[8].x*100)
        f2_y=int(hand_landmarks.landmark[8].y*100)

        cal=(f1_x-f2_x)**2 + (f1_y-f2_y)**2
        dist=sqrt(cal)
        mouse_drag.position=((100-f2_x)*16,f2_y*9) # 모니터 16:9

        if dist<3:  # 집게와 엄지손가락이 붙으면
          mouse_drag.press(mouse_button.left)
          mouse_drag.release(mouse_button.left)
          
    # 좌우반전
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
