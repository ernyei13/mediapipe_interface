import os
import cv2
import json
import base64
import math
import numpy as np
import mediapipe as mp
import pygame
from openai import OpenAI
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Internal Resolution (AI/MediaPipe) and Display Window constants
RES_W, RES_H = 1920, 1080
DISP_W, DISP_H = 1280, 720
SCALE_X = RES_W / DISP_W
SCALE_Y = RES_H / DISP_H

# --- GPT VISION ANALYSIS ---
def analyze_interface(image_path, target_width, target_height):
    """Sends the interface to GPT-4o for coordinate analysis."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = f"""
    Analyze this interface drawing. Return a JSON object with coordinates scaled 
    to a width of {target_width} and height of {target_height}.
    1. 'sliders': Vertical paths. Return {{x, y, w, h}}.
    2. 'buttons': Box shapes. Return {{x, y, w, h}}.
    3. 'knobs': Circle shapes. Return {{x, y, r}}.
    Format: {{ "sliders": [], "buttons": [], "knobs": [] }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}],
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)

# --- PYGAME DRAWING APP ---
def run_sketchpad():
    pygame.init()
    screen = pygame.display.set_mode((DISP_W, DISP_H))
    pygame.display.set_caption("Draw Your Interface (1080p Internal)")
    
    canvas = pygame.Surface((RES_W, RES_H))
    canvas.fill((255, 255, 255))
    
    font = pygame.font.SysFont("Arial", 24, bold=True)
    drawing = False
    last_pos = None
    running = True
    saved_file = "custom_interface.png"

    while running:
        screen.fill((220, 220, 220))
        button_rect = pygame.Rect(DISP_W // 2 - 100, DISP_H - 60, 200, 40)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            
            # Map mouse to high-res coordinates
            m_pos = event.pos if hasattr(event, 'pos') else (0,0)
            canvas_pos = (int(m_pos[0] * SCALE_X), int(m_pos[1] * SCALE_Y))

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    pygame.image.save(canvas, saved_file)
                    running = False 
                elif event.pos[1] < DISP_H - 80:
                    drawing = True
                    last_pos = canvas_pos
            if event.type == pygame.MOUSEBUTTONUP: drawing = False
            if event.type == pygame.MOUSEMOTION and drawing:
                if canvas_pos[1] < RES_H - (80 * SCALE_Y):
                    if last_pos: pygame.draw.line(canvas, (0, 0, 0), last_pos, canvas_pos, 8)
                    last_pos = canvas_pos

        display_canvas = pygame.transform.smoothscale(canvas, (DISP_W, DISP_H))
        screen.blit(display_canvas, (0, 0))
        pygame.draw.rect(screen, (50, 50, 50), button_rect, border_radius=8)
        text_surf = font.render("TRY INTERFACE", True, (255, 255, 255))
        screen.blit(text_surf, (button_rect.x + 20, button_rect.y + 5))
        pygame.display.flip()
        
    pygame.quit()
    return saved_file

# --- MEDIAPIPE APP ---
def start_mediapipe_app(interface_file):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
    
    raw_ui = cv2.imread(interface_file)
    raw_ui = cv2.resize(raw_ui, (RES_W, RES_H))

    ui_data = analyze_interface(interface_file, RES_W, RES_H)
    
    sliders = [{'rect': (s['x'], s['y'], s['w'], s['h']), 'val': 0.5, 'id': i} for i, s in enumerate(ui_data.get('sliders', []))]
    buttons = [{'rect': (b['x'], b['y'], b['w'], b['h']), 'on': False, 'cd': 0} for b in ui_data.get('buttons', [])]
    knobs = [{'center': (k['x'], k['y']), 'radius': k.get('r', 30), 'angle': 0, 'active': False, 'cd': 0, 'id': i} for i, k in enumerate(ui_data.get('knobs', []))]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    active_slider_id = None
    active_knob_id = None

    

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (RES_W, RES_H))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display = cv2.addWeighted(frame, 0.7, raw_ui, 0.3, 0)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                t, idx = hand_lms.landmark[4], hand_lms.landmark[8]
                # Interaction math MUST use high-res RES_W/RES_H
                px, py = int(idx.x * RES_W), int(idx.y * RES_H)
                pinched = math.sqrt((idx.x - t.x)**2 + (idx.y - t.y)**2) < 0.05

                if pinched:
                    # SLIDER INTERACTION
                    if active_slider_id is None:
                        for s in sliders:
                            sx, sy, sw, sh = s['rect']
                            # Increased grab hitbox for high-res
                            if sx - 60 < px < sx + sw + 60 and sy < py < sy + sh:
                                active_slider_id = s['id']; break
                    if active_slider_id is not None:
                        s = sliders[active_slider_id]
                        s['val'] = 1.0 - np.clip((py - s['rect'][1]) / s['rect'][3], 0.0, 1.0)
                    
                    # BUTTON INTERACTION
                    for b in buttons:
                        bx, by, bw, bh = b['rect']
                        if bx < px < bx + bw and by < py < by + bh and b['cd'] == 0:
                            b['on'] = not b['on']; b['cd'] = 25
                    
                    # KNOB INTERACTION (Pinch toggle to tune)
                    for k in knobs:
                        kx, ky = k['center']
                        if math.sqrt((px - kx)**2 + (py - ky)**2) < k['radius'] + 50 and k['cd'] == 0:
                            if active_knob_id == k['id']: 
                                active_knob_id = None; k['active'] = False
                            elif active_knob_id is None: 
                                active_knob_id = k['id']; k['active'] = True
                            k['cd'] = 30
                else:
                    active_slider_id = None
                
                if active_knob_id is not None:
                    k = knobs[active_knob_id]
                    k['angle'] = math.degrees(math.atan2(py - k['center'][1], px - k['center'][0]))

        # SCALE DISPLAY FOR VIEWING
        final_view = cv2.resize(display, (DISP_W, DISP_H))
        
        # RENDER FEEDBACK
        for b in buttons:
            b['cd'] = max(0, b['cd'] - 1)
            bx, by = int(b['rect'][0] / SCALE_X), int(b['rect'][1] / SCALE_Y)
            bw, bh = int(b['rect'][2] / SCALE_X), int(b['rect'][3] / SCALE_Y)
            cv2.rectangle(final_view, (bx, by), (bx+bw, by+bh), (0, 255, 0) if b['on'] else (0, 0, 255), 2)
        
        for s in sliders:
            sx, sy = int((s['rect'][0] + s['rect'][2]//2) / SCALE_X), int((s['rect'][1] + (1.0 - s['val']) * s['rect'][3]) / SCALE_Y)
            cv2.circle(final_view, (sx, sy), 15, (0, 255, 255), -1)

        for k in knobs:
            k['cd'] = max(0, k['cd'] - 1)
            kx, ky = int(k['center'][0] / SCALE_X), int(k['center'][1] / SCALE_Y)
            kr = int(k['radius'] / SCALE_X)
            rad = math.radians(k['angle'])
            cv2.circle(final_view, (kx, ky), kr, (0, 255, 0) if k['active'] else (255, 255, 0), 2)
            cv2.line(final_view, (kx, ky), (int(kx + kr*math.cos(rad)), int(ky + kr*math.sin(rad))), (0, 255, 255), 3)

        cv2.imshow('Hand Interface (Fixed Scaling)', final_view)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Main Menu:\n1. Use Preset\n2. Draw New")
    choice = input("Select: ")
    if choice == "1":
        files = [f for f in os.listdir() if f.endswith(".png")]
        for i, f in enumerate(files): print(f"{i+1}. {f}")
        start_mediapipe_app(files[int(input("Number: ")) - 1])
    else:
        start_mediapipe_app(run_sketchpad())