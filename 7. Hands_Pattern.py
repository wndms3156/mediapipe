# Hands_Pattern.py

import cv2
import mediapipe as mp
import math as mt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
tips=[4,8,12,16,20] 
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pattern = []

#비밀 패턴 설정
pin = [[0,0],[1,0],[1,1],[2,1]]
key=0

while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (800, 600))
    if not success:
        print("Ignoring empty camera frame.")
        break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        posList=[]
        cnt=0
        direction=0

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            pos=hand_landmarks.landmark
            jibgae_x=pos[8].x*800
            jibgae_y=pos[8].y*600
            jungji_x=pos[12].x*800
            jungji_y=pos[12].y*600
            
            # 초기화 버튼 위치
            ini_pattern_x=600  
            ini_pattern_y=150
            
            # 패턴 점 생성
            for i in range(3):
                for j in range(3):
                    x=250+100*j
                    y=150+100*i

                    if mt.sqrt((jibgae_x-x)**2+(jibgae_y-y)**2) <= 20:
                        cv2.circle(image, (x,y), 20, (0,0,255), -1)
                        if len(pattern)==0: pattern.append([i,j])
                        elif pattern[-1] != [i,j]: pattern.append([i,j])
                    else: cv2.circle(image, (x,y), 20, (217,65,197), -1)

            if pin == pattern:
                key=1
                print("open")
                

            #pin 초기화 버튼 
            if mt.sqrt((jibgae_x-ini_pattern_x)**2+(jibgae_y-ini_pattern_y)**2) <= 20:
                cv2.circle(image, (ini_pattern_x,ini_pattern_y), 20, (255,0,0), -1)
                pattern=[]
                key=0
            else:
                cv2.circle(image, (ini_pattern_x,ini_pattern_y), 20, (0,0,255), -1)

            print(pattern)

    image = cv2.flip(image, 1)
    font=cv2.FONT_HERSHEY_SIMPLEX
    if key: cv2.putText(image,"OPEN",(60 ,400),font,8,(0,94,255),20,cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands',image)
    if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()
