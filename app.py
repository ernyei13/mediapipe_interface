import cv2
import mediapipe as mp
import numpy as np
import os

# --- 1. INITIALIZATION & UI DETECTION ---
INTERFACE_PATH = "interface.png" # Your drawn picture
if not os.path.exists(INTERFACE_PATH):
    bg_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(bg_img, "Please place interface.png in folder", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    buttons = []
    sliders = []
else:
    bg_img = cv2.imread(INTERFACE_PATH)
    bg_img = cv2.resize(bg_img, (640, 480))
    
    # Process the drawing to find interactive elements
    gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buttons = [] # Store {'rect': (x,y,w,h), 'state': False, 'cooldown': 0}
    sliders = [] # Store {'rect': (x,y,w,h), 'val': 50}

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15: continue
        
        # Geometry classification
        if h > w * 1.8: 
            sliders.append({'rect': (x, y, w, h), 'val': 50})
        else:
            buttons.append({'rect': (x, y, w, h), 'state': False, 'cooldown': 0})

# Sort: Sliders Left-to-Right, Buttons Top-to-Bottom
sliders = sorted(sliders, key=lambda s: s['rect'][0])
buttons = sorted(buttons, key=lambda b: (b['rect'][1], b['rect'][0]))

# --- 2. MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

def is_pinching(thumb, index):
    # Calculate Euclidean distance
    dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
    return dist < 0.05

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Display: Camera blended with UI image
    display = cv2.addWeighted(frame, 0.6, bg_img, 0.4, 0)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            t = hand_landmarks.landmark[4]
            idx = hand_landmarks.landmark[8]
            px, py = int(idx.x * 640), int(idx.y * 480)

            if is_pinching(t, idx):
                # Pinch Feedback
                cv2.circle(display, (px, py), 10, (255, 255, 255), 2)
                
                # --- INTERACTION LOGIC ---
                # 1. Update Sliders
                for s in sliders:
                    sx, sy, sw, sh = s['rect']
                    if sx - 30 < px < sx + sw + 30 and sy < py < sy + sh:
                        s['val'] = int(np.clip((1 - (py - sy)/sh) * 100, 0, 100))
                
                # 2. Update Buttons
                for b in buttons:
                    bx, by, bw, bh = b['rect']
                    if bx < px < bx + bw and by < py < by + bh:
                        if b['cooldown'] == 0:
                            b['state'] = not b['state']
                            b['cooldown'] = 20 # Prevent rapid flickering

    # --- 3. DRAW DYNAMIC OVERLAYS ---
    # Render Sliders: Rectangle handles on the slider lines
    for s in sliders:
        sx, sy, sw, sh = s['rect']
        # Calculate vertical position based on value
        val_y = sy + sh - int((s['val']/100) * sh)
        cv2.rectangle(display, (sx-5, val_y-10), (sx+sw+5, val_y+10), (0, 255, 255), -1)
        cv2.putText(display, f"{s['val']}%", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Render Buttons: Glowing circles inside your drawn boxes
    for b in buttons:
        b['cooldown'] = max(0, b['cooldown'] - 1)
        bx, by, bw, bh = b['rect']
        color = (0, 255, 0) if b['state'] else (0, 0, 255)
        # Place circle in center of detected box
        cv2.circle(display, (bx + bw//2, by + bh//2), min(bw, bh)//3, color, -1)

    cv2.imshow('Custom Drawn Interface Controller', display)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()