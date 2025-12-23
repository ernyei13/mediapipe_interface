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

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
# Using GPT-5.2 Pro for superior vision reasoning
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5.2" 

RES_W, RES_H = 1920, 1080
DISP_W, DISP_H = 1280, 720
SCALE_X = RES_W / DISP_W
SCALE_Y = RES_H / DISP_H
INTERFACE_FILE = "custom_interface.png"

def analyze_interface(image_path, target_width, target_height):
    """Sends drawing to GPT-5.2 for high-precision coordinate analysis."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = f"""
    Analyze this interface drawing with high precision. 
    Return a JSON object with coordinates scaled to {target_width}x{target_height}.
    
    1. 'sliders': Vertical tracking lines. Return {{x, y, w, h}}.
    2. 'buttons': Interactive box shapes. Return {{x, y, w, h}}.
    3. 'knobs': Circular controls. Return {{x, y, r}}.
    
    Format your response as a valid JSON object:
    {{ "sliders": [], "buttons": [], "knobs": [] }}
    """
    
    # Utilizing GPT-5.2's improved JSON mode and vision capabilities
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt}, 
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }],
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)

# --- 2. ADVANCED SKETCHPAD ---
def run_sketchpad(existing_file=None):
    pygame.init()
    screen = pygame.display.set_mode((DISP_W, DISP_H))
    pygame.display.set_caption("Sketchpad - Advanced Edit Mode")
    
    canvas = pygame.Surface((RES_W, RES_H))
    undo_stack = [] 

    if existing_file and os.path.exists(existing_file):
        try:
            loaded_img = pygame.image.load(existing_file)
            canvas.blit(pygame.transform.scale(loaded_img, (RES_W, RES_H)), (0, 0))
        except:
            canvas.fill((255, 255, 255))
    else:
        canvas.fill((255, 255, 255))
    
    undo_stack.append(canvas.copy())
    font = pygame.font.SysFont("Arial", 18, bold=True)
    drawing, running, last_pos = False, True, None
    mode = "PEN" 

    while running:
        screen.fill((200, 200, 200))
        btn_try = pygame.Rect(DISP_W - 160, DISP_H - 50, 140, 40)
        btn_undo = pygame.Rect(20, DISP_H - 50, 90, 40)
        btn_clear = pygame.Rect(120, DISP_H - 50, 90, 40)
        btn_pen = pygame.Rect(220, DISP_H - 50, 90, 40)
        btn_erase = pygame.Rect(320, DISP_H - 50, 90, 40)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            m_pos = event.pos if hasattr(event, 'pos') else (0,0)
            canvas_pos = (int(m_pos[0] * SCALE_X), int(m_pos[1] * SCALE_Y))

            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_try.collidepoint(event.pos):
                    pygame.image.save(canvas, INTERFACE_FILE)
                    running = False 
                elif btn_undo.collidepoint(event.pos):
                    if len(undo_stack) > 1:
                        undo_stack.pop() 
                        canvas = undo_stack[-1].copy() 
                elif btn_clear.collidepoint(event.pos):
                    undo_stack.append(canvas.copy())
                    canvas.fill((255, 255, 255))
                elif btn_pen.collidepoint(event.pos): mode = "PEN"
                elif btn_erase.collidepoint(event.pos): mode = "ERASER"
                elif event.pos[1] < DISP_H - 60:
                    undo_stack.append(canvas.copy()) 
                    drawing, last_pos = True, canvas_pos

            if event.type == pygame.MOUSEBUTTONUP: drawing = False
            if event.type == pygame.MOUSEMOTION and drawing:
                if canvas_pos[1] < RES_H - (60 * SCALE_Y):
                    color = (0, 0, 0) if mode == "PEN" else (255, 255, 255)
                    thickness = 8 if mode == "PEN" else 60
                    if last_pos: pygame.draw.line(canvas, color, last_pos, canvas_pos, thickness)
                    last_pos = canvas_pos

        display_canvas = pygame.transform.smoothscale(canvas, (DISP_W, DISP_H))
        screen.blit(display_canvas, (0, 0))
        
        pygame.draw.rect(screen, (40, 40, 180), btn_try); screen.blit(font.render("TRY INTERFACE", True, (255,255,255)), (btn_try.x+10, btn_try.y+11))
        pygame.draw.rect(screen, (80, 80, 80), btn_undo); screen.blit(font.render("UNDO", True, (255,255,255)), (btn_undo.x+22, btn_undo.y+11))
        pygame.draw.rect(screen, (150, 50, 50), btn_clear); screen.blit(font.render("CLEAR", True, (255,255,255)), (btn_clear.x+20, btn_clear.y+11))
        
        p_col = (50, 180, 50) if mode == "PEN" else (100, 100, 100)
        e_col = (50, 180, 50) if mode == "ERASER" else (100, 100, 100)
        pygame.draw.rect(screen, p_col, btn_pen); screen.blit(font.render("PEN", True, (255,255,255)), (btn_pen.x+30, btn_pen.y+11))
        pygame.draw.rect(screen, e_col, btn_erase); screen.blit(font.render("RUBBER", True, (255,255,255)), (btn_erase.x+15, btn_erase.y+11))
        
        pygame.display.flip()
    pygame.quit()
    return INTERFACE_FILE

# --- 3. INTERACTIVE MEDIAPIPE APP ---
def start_mediapipe_app(interface_file):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
    
    raw_ui = cv2.imread(interface_file)
    if raw_ui is None: return "QUIT"
    raw_ui = cv2.resize(raw_ui, (RES_W, RES_H))

    ui_data = analyze_interface(interface_file, RES_W, RES_H)
    
    sliders = [{'rect': (s['x'], s['y'], s['w'], s['h']), 'val': 0.5, 'id': i} for i, s in enumerate(ui_data.get('sliders', []))]
    buttons = [{'rect': (b['x'], b['y'], b['w'], b['h']), 'on': False, 'cd': 0} for b in ui_data.get('buttons', [])]
    knobs = [{'center': (k['x'], k['y']), 'radius': k.get('r', 30), 'angle': 0, 'active': False, 'cd': 0, 'id': i} for i, k in enumerate(ui_data.get('knobs', []))]

    edit_btn_rect = (RES_W - 220, 20, 200, 80)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
    
    active_slider_id, active_knob_id = None, None

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (RES_W, RES_H))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display = cv2.addWeighted(frame, 0.7, raw_ui, 0.3, 0)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                t = hand_lms.landmark[mp_hands.HandLandmark.THUMB_TIP] 
                idx = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
                px, py = int(idx.x * RES_W), int(idx.y * RES_H)
                pinched = math.sqrt((idx.x - t.x)**2 + (idx.y - t.y)**2) < 0.05

                if pinched:
                    ex, ey, ew, eh = edit_btn_rect
                    if ex < px < ex + ew and ey < py < ey + eh:
                        cap.release(); cv2.destroyAllWindows(); return "EDIT"

                    if active_slider_id is None:
                        for s in sliders:
                            sx, sy, sw, sh = s['rect']
                            if sx - 60 < px < sx + sw + 60 and sy < py < sy + sh:
                                active_slider_id = s['id']; break
                    if active_slider_id is not None:
                        s = sliders[active_slider_id]
                        s['val'] = 1.0 - np.clip((py - s['rect'][1]) / s['rect'][3], 0.0, 1.0)
                    for b in buttons:
                        bx, by, bw, bh = b['rect']
                        if bx < px < bx + bw and by < py < by + bh and b['cd'] == 0:
                            b['on'] = not b['on']; b['cd'] = 25
                    for k in knobs:
                        kx, ky = k['center']
                        if math.sqrt((px - kx)**2 + (py - ky)**2) < k['radius'] + 50 and k['cd'] == 0:
                            if active_knob_id == k['id']: active_knob_id = None; k['active'] = False
                            elif active_knob_id is None: active_knob_id = k['id']; k['active'] = True
                            k['cd'] = 30
                else: active_slider_id = None
                if active_knob_id is not None:
                    knob = knobs[active_knob_id]
                    knob['angle'] = math.degrees(math.atan2(py - knob['center'][1], px - knob['center'][0]))

        final_view = cv2.resize(display, (DISP_W, DISP_H))
        
        ev_x, ev_y = int(edit_btn_rect[0]/SCALE_X), int(edit_btn_rect[1]/SCALE_Y)
        ev_w, ev_h = int(edit_btn_rect[2]/SCALE_X), int(edit_btn_rect[3]/SCALE_Y)
        cv2.rectangle(final_view, (ev_x, ev_y), (ev_x+ev_w, ev_y+ev_h), (180, 40, 40), -1)
        cv2.putText(final_view, "EDIT", (ev_x+55, ev_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        for b in buttons:
            b['cd'] = max(0, b['cd'] - 1)
            bx, by = int(b['rect'][0] / SCALE_X), int(b['rect'][1] / SCALE_Y)
            bw, bh = int(b['rect'][2] / SCALE_X), int(b['rect'][3] / SCALE_Y)
            cv2.rectangle(final_view, (bx, by), (bx+bw, by+bh), (0, 255, 0) if b['on'] else (0, 0, 255), 2)
            cv2.putText(final_view, f"{1 if b['on'] else 0}", (bx + bw//2 - 5, by + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for s in sliders:
            sx, sy = int((s['rect'][0] + s['rect'][2]//2) / SCALE_X), int((s['rect'][1] + (1.0 - s['val']) * s['rect'][3]) / SCALE_Y)
            cv2.circle(final_view, (sx, sy), 15, (0, 255, 255), -1)
            base_x, base_y = int(s['rect'][0] / SCALE_X), int((s['rect'][1] + s['rect'][3]) / SCALE_Y)
            cv2.putText(final_view, f"{int(s['val'] * 100)}", (base_x, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for k in knobs:
            k['cd'] = max(0, k['cd'] - 1)
            kx, ky = int(k['center'][0] / SCALE_X), int(k['center'][1] / SCALE_Y)
            kr, rad = int(k['radius'] / SCALE_X), math.radians(k['angle'])
            cv2.circle(final_view, (kx, ky), kr, (0, 255, 0) if k['active'] else (255, 255, 0), 2)
            cv2.line(final_view, (kx, ky), (int(kx + kr*math.cos(rad)), int(ky + kr*math.sin(rad))), (0, 255, 255), 3)
            norm_angle = (k['angle'] + 180) % 360
            cv2.putText(final_view, f"{int((norm_angle / 360) * 100)}", (kx - 10, ky + kr + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('Hand Interface', final_view)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows(); return "QUIT"

# --- 4. MAIN FLOW ---
if __name__ == "__main__":
    current_file = None
    print("1: Preset, 2: New Drawing")
    choice = input("Select: ")
    if choice == "1":
        files = [f for f in os.listdir() if f.endswith(".png")]
        if not files:
            current_file = run_sketchpad()
        else:
            for i, f in enumerate(files): print(f"{i+1}. {f}")
            current_file = files[int(input("Number: ")) - 1]
    else:
        current_file = run_sketchpad()

    while True:
        status = start_mediapipe_app(current_file)
        if status == "EDIT": current_file = run_sketchpad(existing_file=current_file)
        else: break