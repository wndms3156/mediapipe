# Hands_Counter.py

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 웹캠용
cap = cv2.VideoCapture(0)
tips=[4,8,12,16,20]  # 손가락 끝
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 손가락 모양대로 표시
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    flag=0  
    if results.multi_hand_landmarks:
      flag=1
      posList=[]
      cnt=0
      direction=[0,0]
      for i in range(len(results.multi_hand_landmarks)):
        hand_landmarks=results.multi_hand_landmarks[i]
        #print(hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        pos=hand_landmarks.landmark
        if pos[tips[0]].x*(1-direction[i]*0.5) < pos[tips[4]].x*(1-(1-direction[i])*0.1):
            direction[i]=1
            
        #print(direction[i])

        # 엄지
        if pos[tips[0]-direction[i]].x > pos[tips[0]-(1-direction[i])].x: cnt+=1

        # 나머지 손가락
        for i in range(1,5):
            if pos[tips[i]].y < pos[tips[i]-2].y: cnt+=1

    # 좌우반전
    image = cv2.flip(image, 1)
    font = cv2.FONT_HERSHEY_PLAIN
    if flag: cv2.putText(image, str(cnt), (45, 375), font, 10, (140, 65, 217), 20)
    cv2.imshow('MediaPipe Hands',image) 
    if cv2.waitKey(5) & 0xFF == 27: break
cap.release()
