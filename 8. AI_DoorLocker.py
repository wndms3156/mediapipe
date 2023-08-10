# AI_DoorLocker.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
from math import sqrt
import mediapipe as mp  
import drawing_utils_adh as mp_drawing 

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
tips=[4,8,12,16,20]
font=cv2.FONT_HERSHEY_SIMPLEX
hands=mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

pattern = []

#초기 비밀 패턴 설정
key_pattern = [[0,0],[1,0],[1,1],[2,1]]
ini_key=0
key=0

#패턴 입력 초기화
ini_pattern_x=650
ini_pattern_y=150

#비밀 패턴 초기화
ini_key_x=650
ini_key_y=250

#비밀 패턴 설정
ok_key_x=650
ok_key_y=350

authority_key=0

while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (800, 600))
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    ini_btn_flag=0
    flag=0 
    if hand_results.multi_hand_landmarks:
        flag=1
        posList=[]
        cnt=0
        direction=0

        for hand_landmarks in hand_results.multi_hand_landmarks:
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

            #패턴 입력 초기화 버튼
            if sqrt((jibgae_x-ini_pattern_x)**2+(jibgae_y-ini_pattern_y)**2) <= 20:
                cv2.circle(image, (ini_pattern_x,ini_pattern_y), 20, (0,94,255), -1)
                pattern=[]
                key=0
            else:
                cv2.circle(image, (ini_pattern_x,ini_pattern_y), 20, (116,116,116), -1)
                
            # 패턴 점 생성
            
            for i in range(3):
                for j in range(3):
                    x=200+100*j
                    y=150+100*i  

                    if sqrt((jibgae_x-x)**2+(jibgae_y-y)**2) <= 20:
                        cv2.circle(image, (x,y), 20, (0,0,255), -1)
                        if len(pattern)==0: pattern.append([i,j])
                        elif pattern[len(pattern)-1] != [i,j]: pattern.append([i,j])
                    else: cv2.circle(image, (x,y), 20, (217,65,197), -1)

            if ini_key and pattern:
                for i in range(len(pattern)):
                    x,y=pattern[i]
                    cv2.circle(image, (200+100*y,150+100*x), 20, (0,0,255-i*25), -1)

            if pattern and key_pattern and key_pattern == pattern:
                key=1
                authority_key=1
                print("open")
 
            #비밀 패턴 재설정
            if authority_key:
                if sqrt((jibgae_x-ini_key_x)**2+(jibgae_y-ini_key_y)**2) <= 20:
                    cv2.circle(image, (ini_key_x,ini_key_y), 20, (0,94,255), -1)
                    pattern=[]
                    key_pattern=[]
                    ini_key=1
                    key=0
                else:
                    cv2.circle(image, (ini_key_x,ini_key_y), 20, (116,116,116), -1)
            #비밀 패턴 저장
                if sqrt((jibgae_x-ok_key_x)**2+(jibgae_y-ok_key_y)**2) <= 20:
                    cv2.circle(image, (ok_key_x,ok_key_y), 20, (0,94,255), -1)
                    if len(key_pattern)==0:
                        key_pattern = pattern
                        pattern=[]
                    ini_key=0
                else:
                    cv2.circle(image, (ok_key_x,ok_key_y), 20, (80,80,80), -1)
                print(f"pattern:{pattern}")    
                print(f"key_pattern:{key_pattern}")

    image = cv2.flip(image, 1)
    if key: cv2.putText(image,"OPEN",(60 ,400),font,8,(0,94,255),20,cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
