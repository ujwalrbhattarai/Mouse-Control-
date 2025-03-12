import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# Set pyautogui parameters
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Get screen size
SCREEN_W, SCREEN_H = pyautogui.size()
print(f"Screen resolution: {SCREEN_W}x{SCREEN_H}")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Verify actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# Smoothing buffer
POSITION_BUFFER = deque(maxlen=5)

# Click and Scroll Thresholds
SCROLL_THRESHOLD = 0.02
CLICK_DISTANCE_THRESHOLD = 0.05  # For detecting closed palm
CLICK_DELAY = 1  # 1 sec delay
last_click_time = 0

# Last Y position for scrolling
last_scroll_y = None  

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        print("Frame capture failed. Retrying...")
        continue  

    # Flip and convert image
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Get image dimensions
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get fingertips
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Raw normalized coordinates
        raw_x, raw_y = index_tip.x, index_tip.y

        # Smooth coordinates
        POSITION_BUFFER.append((raw_x, raw_y))
        smoothed_x = np.mean([pos[0] for pos in POSITION_BUFFER])
        smoothed_y = np.mean([pos[1] for pos in POSITION_BUFFER])

        # Map to screen coordinates
        screen_x = np.interp(smoothed_x, [0.05, 0.95], [0, SCREEN_W])
        screen_y = np.interp(smoothed_y, [0.05, 0.95], [0, SCREEN_H])

        # Clamp values to screen boundaries
        screen_x = max(1, min(screen_x, SCREEN_W - 2))
        screen_y = max(1, min(screen_y, SCREEN_H - 2))

        # Move cursor
        pyautogui.moveTo(screen_x, screen_y)

        # Click detection: Check if index & pinky are close together
        distance = calculate_distance(index_tip, pinky_tip)
        current_time = time.time()
        if distance < CLICK_DISTANCE_THRESHOLD and (current_time - last_click_time) > CLICK_DELAY:
            pyautogui.click()
            last_click_time = current_time
            print("Click!")

        # Scrolling logic: Only scroll when index & middle fingers move UP/DOWN
        scroll_y = (index_tip.y + middle_tip.y) / 2  # Average Y position of index & middle finger

        if last_scroll_y is not None:
            if scroll_y < last_scroll_y - SCROLL_THRESHOLD:  # Fingers moved UP → Scroll Up
                pyautogui.scroll(10)
                print("Scrolling Up")
            elif scroll_y > last_scroll_y + SCROLL_THRESHOLD:  # Fingers moved DOWN → Scroll Down
                pyautogui.scroll(-10)
                print("Scrolling Down")

        last_scroll_y = scroll_y  # Update last known Y position

        # Draw landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show image
    cv2.imshow('Hand Gesture Control', image)

    # Small delay to prevent CPU overload and flickering
    time.sleep(0.01)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
