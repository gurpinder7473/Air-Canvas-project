import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
color_index = 0
brush_color = colors[color_index]
eraser_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x1 = int(hand_landmarks.landmark[8].x * w)
            y1 = int(hand_landmarks.landmark[8].y * h)

            x2 = int(hand_landmarks.landmark[12].x * w)
            y2 = int(hand_landmarks.landmark[12].y * h)

            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)

            fingers = []
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                fingers.append(1)
            else:
                fingers.append(0)

            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                fingers.append(1)
            else:
                fingers.append(0)

            if fingers == [1,1]:
                color_index = (color_index + 1) % len(colors)
                brush_color = colors[color_index]
                cv2.putText(frame, "Color Changed!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, brush_color, 3)

            distance = np.hypot(x1 - x_thumb, y1 - y_thumb)
            if distance < 40:
                eraser_mode = True
                cv2.putText(frame, "Eraser", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            else:
                eraser_mode = False

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            if fingers == [1,0]:
                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), (0,0,0), 40)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), brush_color, 5)

            prev_x, prev_y = x1, y1
    else:
        prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Air Canvas 2.0", output)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        cv2.imwrite("screenshots/my_drawing.png", canvas)
        print("Drawing saved as screenshots/my_drawing.png")

cap.release()
cv2.destroyAllWindows()
