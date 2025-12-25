import os
import cv2
import json
import base64
import math
import numpy as np
import mediapipe as mp
import pygame
import time
from openai import OpenAI
from dotenv import load_dotenv
import mido

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5.2" 

# Virtual MIDI Port
try:
    midiout = mido.open_output('HandInterfacePort', virtual=True)
except:
    midiout = None # Fallback for OS that doesn't support virtual ports

def send_midi_cc(control, value):
    if midiout:
        msg = mido.Message('control_change', control=control, value=int(value))
        midiout.send(msg)

def send_midi_note(note, state):
    if midiout:
        msg = mido.Message('note_on' if state else 'note_off', note=note, velocity=127)
        midiout.send(msg)

RES_W, RES_H = 1920, 1080
DISP_W, DISP_H = 1280, 720
SCALE_X = RES_W / DISP_W
SCALE_Y = RES_H / DISP_H
INTERFACE_FILE = "custom_interface.png"


def get_transform_matrix(anchors, target_w, target_h):
    # Points on the skewed paper (from GPT)
    src_pts = np.array([ [a['x'], a['y']] for a in anchors ], dtype="float32")
    # Points where they SHOULD be (the corners of your screen)
    dst_pts = np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ], dtype="float32")
    
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def transform_point(point, matrix):
    # Transforms a camera (x, y) point into the "Paper Space" (x, y)
    px, py = point
    new_pt = np.array([[[px, py]]], dtype="float32")
    transformed = cv2.perspectiveTransform(new_pt, matrix)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

def analyze_interface(image_path, target_width, target_height):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = f"""
    Analyze this interface. 
    1. 'anchors': Find the 4 corner triangles. Return them as [top_left, top_right, bottom_right, bottom_left] coordinates.
    2. 'sliders': Vertical lines. Return {{x, y, w, h}}.
    3. 'buttons': Box shapes. If it has an 'X', mark as 'momentary'. Return {{x, y, w, h, type}}.
    4. 'knobs': Circular controls. Return {{x, y, r}}.
    
    IMPORTANT: Provide the most TIGHT bounding boxes possible for accuracy. IT SHOULD BE PIXEL PERFECT TO THE CONTROL ELEMENTS SO IT OVERLAYS NICELY ON THE IMAGE.
    Format as JSON: {{ "anchors": [], "sliders": [], "buttons": [], "knobs": [] }}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}],
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

    PRESETS_DIR = "presets"
    if not os.path.exists(PRESETS_DIR): os.makedirs(PRESETS_DIR)
    
    def get_presets(): return [f for f in os.listdir(PRESETS_DIR) if f.endswith(".png")]
    preset_files = get_presets()

    if existing_file and os.path.exists(existing_file):
        try:
            loaded_img = pygame.image.load(existing_file)
            canvas.blit(pygame.transform.scale(loaded_img, (RES_W, RES_H)), (0, 0))
        except: canvas.fill((255, 255, 255))
    else: canvas.fill((255, 255, 255))
    
    undo_stack.append(canvas.copy())
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 14)
    drawing, running, last_pos, mode = False, True, None, "PEN" 
    dropdown_open, saving_preset, preset_name = False, False, ""
    selection_start = None
    selection_rect = pygame.Rect(0, 0, 0, 0) 
    clipboard_surf, is_selecting = None, False

    while running:
        screen.fill((200, 200, 200))
        keys = pygame.key.get_pressed()
        ctrl_cmd = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_LMETA] or keys[pygame.K_RMETA]

        btn_try = pygame.Rect(DISP_W - 140, DISP_H - 50, 120, 40)
        btn_undo = pygame.Rect(10, DISP_H - 50, 70, 40)
        btn_clear = pygame.Rect(90, DISP_H - 50, 70, 40)
        btn_save_pre = pygame.Rect(170, DISP_H - 50, 110, 40)
        btn_presets = pygame.Rect(290, DISP_H - 50, 100, 40)
        btn_pen = pygame.Rect(400, DISP_H - 50, 60, 40)
        btn_erase = pygame.Rect(470, DISP_H - 50, 80, 40)
        btn_select = pygame.Rect(560, DISP_H - 50, 80, 40)
        btn_paste = pygame.Rect(650, DISP_H - 50, 80, 40)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            if event.type == pygame.KEYDOWN and not saving_preset:
                if ctrl_cmd:
                    if event.key == pygame.K_z and len(undo_stack) > 1:
                        undo_stack.pop(); canvas = undo_stack[-1].copy()
                    elif event.key == pygame.K_k:
                        undo_stack.append(canvas.copy()); canvas.fill((255, 255, 255))
                    elif event.key == pygame.K_c and selection_rect and selection_rect.width > 5:
                        clipboard_surf = canvas.subsurface(selection_rect.normalize()).copy()
                    elif event.key == pygame.K_v and clipboard_surf: mode = "PASTE"

            if saving_preset:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if preset_name:
                            pygame.image.save(canvas, os.path.join(PRESETS_DIR, f"{preset_name}.png"))
                            preset_files = get_presets()
                        saving_preset, preset_name = False, ""
                    elif event.key == pygame.K_BACKSPACE: preset_name = preset_name[:-1]
                    else: preset_name += event.unicode
                continue

            m_pos = event.pos if hasattr(event, 'pos') else (0,0)
            canvas_pos = (int(m_pos[0] * SCALE_X), int(m_pos[1] * SCALE_Y))

            if event.type == pygame.MOUSEBUTTONDOWN:
                if dropdown_open and event.button == 3:
                    for i, file in enumerate(preset_files):
                        item_rect = pygame.Rect(290, DISP_H - 95 - (i * 30), 100, 30)
                        if item_rect.collidepoint(event.pos):
                            os.remove(os.path.join(PRESETS_DIR, file)); preset_files = get_presets(); break
                if event.button == 1:
                    if dropdown_open:
                        item_clicked = False
                        for i, file in enumerate(preset_files):
                            item_rect = pygame.Rect(290, DISP_H - 95 - (i * 30), 100, 30)
                            if item_rect.collidepoint(event.pos):
                                undo_stack.append(canvas.copy())
                                img = pygame.image.load(os.path.join(PRESETS_DIR, file))
                                canvas.blit(pygame.transform.scale(img, (RES_W, RES_H)), (0, 0))
                                dropdown_open = False; item_clicked = True; break
                        if not item_clicked and not btn_presets.collidepoint(event.pos): dropdown_open = False
                    if btn_try.collidepoint(event.pos): pygame.image.save(canvas, INTERFACE_FILE); running = False 
                    elif btn_presets.collidepoint(event.pos): dropdown_open = not dropdown_open
                    elif btn_save_pre.collidepoint(event.pos): saving_preset = True
                    elif btn_undo.collidepoint(event.pos):
                        if len(undo_stack) > 1: undo_stack.pop(); canvas = undo_stack[-1].copy() 
                    elif btn_clear.collidepoint(event.pos): undo_stack.append(canvas.copy()); canvas.fill((255, 255, 255))
                    elif btn_pen.collidepoint(event.pos): mode = "PEN"
                    elif btn_erase.collidepoint(event.pos): mode = "ERASER"
                    elif btn_select.collidepoint(event.pos): mode = "SELECT"
                    elif btn_paste.collidepoint(event.pos): 
                        if clipboard_surf: mode = "PASTE"
                    elif event.pos[1] < DISP_H - 60:
                        if mode == "SELECT":
                            is_selecting, selection_start = True, canvas_pos
                            selection_rect = pygame.Rect(canvas_pos, (1, 1))
                        elif mode == "PASTE" and clipboard_surf:
                            undo_stack.append(canvas.copy())
                            canvas.blit(clipboard_surf, canvas_pos); mode = "PEN"
                        else:
                            undo_stack.append(canvas.copy()); drawing, last_pos = True, canvas_pos

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                if mode == "SELECT" and is_selecting:
                    is_selecting = False
                    if selection_rect is not None:
                        final_rect = selection_rect.normalize()
                        if final_rect.width > 5 and final_rect.height > 5:
                            try: clipboard_surf = canvas.subsurface(final_rect).copy(); mode = "PASTE"
                            except: pass
                selection_start = None

            if event.type == pygame.MOUSEMOTION:
                if is_selecting and selection_start:
                    cx, cy = canvas_pos; sx, sy = selection_start
                    selection_rect = pygame.Rect(min(sx, cx), min(sy, cy), abs(cx - sx), abs(cy - sy))
                elif drawing:
                    color, thick = ((0, 0, 0), 8) if mode == "PEN" else ((255, 255, 255), 60)
                    if last_pos: pygame.draw.line(canvas, color, last_pos, canvas_pos, thick)
                    last_pos = canvas_pos

        display_canvas = pygame.transform.smoothscale(canvas, (DISP_W, DISP_H))
        screen.blit(display_canvas, (0, 0))
        if is_selecting and selection_rect:
            disp_rect = pygame.Rect(selection_rect.x / SCALE_X, selection_rect.y / SCALE_Y, selection_rect.width / SCALE_X, selection_rect.height / SCALE_Y)
            pygame.draw.rect(screen, (0, 120, 255), disp_rect, 2)
        if mode == "PASTE" and clipboard_surf:
            mx, my = pygame.mouse.get_pos()
            if my < DISP_H - 60:
                ghost = pygame.transform.smoothscale(clipboard_surf, (int(clipboard_surf.get_width()/SCALE_X), int(clipboard_surf.get_height()/SCALE_Y)))
                ghost.set_alpha(150); screen.blit(ghost, (mx, my))

        pygame.draw.rect(screen, (40, 40, 180), btn_try); screen.blit(font.render("TRY", True, (255,255,255)), (btn_try.x+45, btn_try.y+11))
        pygame.draw.rect(screen, (80, 80, 80), btn_undo); screen.blit(font.render("UNDO", True, (255,255,255)), (btn_undo.x+10, btn_undo.y+11))
        pygame.draw.rect(screen, (150, 50, 50), btn_clear); screen.blit(font.render("CLR", True, (255,255,255)), (btn_clear.x+15, btn_clear.y+11))
        pygame.draw.rect(screen, (60, 120, 60), btn_save_pre); screen.blit(font.render("SAVE PRE", True, (255,255,255)), (btn_save_pre.x+15, btn_save_pre.y+11))
        pygame.draw.rect(screen, (60, 60, 60), btn_presets); screen.blit(font.render("PRESETS", True, (255,255,255)), (btn_presets.x+10, btn_presets.y+11))
        for b, m, t in [(btn_pen, "PEN", "PEN"), (btn_erase, "ERASER", "RUBBER"), (btn_select, "SELECT", "SELECT"), (btn_paste, "PASTE", "PASTE")]:
            pygame.draw.rect(screen, (50, 180, 50) if mode == m else (100, 100, 100), b)
            screen.blit(font.render(t, True, (255,255,255)), (b.x+10, b.y+11))
        
        if dropdown_open:
            for i, file in enumerate(preset_files):
                item_rect = pygame.Rect(290, DISP_H - 95 - (i * 30), 100, 30)
                pygame.draw.rect(screen, (240, 240, 240), item_rect); pygame.draw.rect(screen, (0, 0, 0), item_rect, 1)
                screen.blit(small_font.render(file[:15], True, (0, 0, 0)), (item_rect.x + 5, item_rect.y + 5))
        
        if saving_preset:
            overlay = pygame.Surface((DISP_W, DISP_H), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); screen.blit(overlay, (0, 0))
            pygame.draw.rect(screen, (255, 255, 255), (DISP_W//2 - 150, DISP_H//2 - 50, 300, 100), border_radius=10)
            screen.blit(font.render("Name your preset:", True, (0, 0, 0)), (DISP_W//2 - 80, DISP_H//2 - 40))
            pygame.draw.rect(screen, (220, 220, 220), (DISP_W//2 - 130, DISP_H//2, 260, 30))
            screen.blit(font.render(preset_name + "|", True, (0, 0, 0)), (DISP_W//2 - 120, DISP_H//2 + 5))
        pygame.display.flip()
    pygame.quit(); return INTERFACE_FILE

# --- 3. MEDIAPIPE APP WITH PAPER CAPTURE ---
def start_mediapipe_app(interface_file):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
    raw_ui = cv2.imread(interface_file)
    if raw_ui is None: return "QUIT"
    raw_ui = cv2.resize(raw_ui, (RES_W, RES_H))
    
    ui_data = analyze_interface(interface_file, RES_W, RES_H)

    sliders = [{'rect': (s['x'], s['y'], s['w'], s['h']), 'val': 0.5, 'id': i} for i, s in enumerate(ui_data.get('sliders', []))]
    buttons = [{'rect': (b['x'], b['y'], b['w'], b['h']), 'on': False, 'cd': 0, 'id': i, 'momentary': b.get('type') == 'momentary'} for i, b in enumerate(ui_data.get('buttons', []))]
    knobs = [{'center': (k['x'], k['y']), 'radius': k.get('r', 30), 'angle': 0, 'id': i} for i, k in enumerate(ui_data.get('knobs', []))]
    
    edit_btn_rect = (RES_W - 220, 20, 200, 80)
    capture_btn_rect = (RES_W - 220, 120, 200, 80) # New Capture button below Edit
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    active_slider_id, active_knob_id = None, None

    

    while cap.isOpened():

        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1) 
        
        # 2. Process the mirrored frame for hand tracking
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame = cv2.flip(frame, 1); frame = cv2.resize(frame, (RES_W, RES_H))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display = cv2.addWeighted(frame, 0.7, raw_ui, 0.3, 0)

        for b in buttons: 
            if b['momentary']: b['on'] = False

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                t, idx = hand_lms.landmark[4], hand_lms.landmark[8]
                px, py = int(idx.x * RES_W), int(idx.y * RES_H)
                pinched = math.sqrt((idx.x - t.x)**2 + (idx.y - t.y)**2) < 0.05
                
                if pinched:
                    # EDIT Trigger
                    ex, ey, ew, eh = edit_btn_rect
                    if ex < px < ex + ew and ey < py < ey + eh:
                        cap.release(); cv2.destroyAllWindows(); return "EDIT"
                    
                    # CAPTURE Trigger
                    cx, cy, cw, ch = capture_btn_rect
                    if cx < px < cx + cw and cy < py < cy + ch:
                        cv2.imwrite(INTERFACE_FILE, frame) # Capture the paper
                        cap.release(); cv2.destroyAllWindows(); return "CAPTURE"
                    
                    if active_slider_id is None:
                        for s in sliders:
                            if s['rect'][0]-60 < px < s['rect'][0]+s['rect'][2]+60 and s['rect'][1] < py < s['rect'][1]+s['rect'][3]:
                                active_slider_id = s['id']; break
                    if active_slider_id is not None:
                        s = next(item for item in sliders if item["id"] == active_slider_id)
                        s['val'] = 1.0 - np.clip((py - s['rect'][1]) / s['rect'][3], 0.0, 1.0)
                        send_midi_cc(control=s['id'], value=s['val'] * 127)
                    
                    if active_knob_id is None:
                        for k in knobs:
                            if math.sqrt((px - k['center'][0])**2 + (py - k['center'][1])**2) < k['radius'] + 50:
                                active_knob_id = k['id']; break
                    if active_knob_id is not None:
                        k = next(item for item in knobs if item["id"] == active_knob_id)
                        k['angle'] = math.degrees(math.atan2(py - k['center'][1], px - k['center'][0]))
                        send_midi_cc(control=k['id'] + 20, value=int(((k['angle'] % 360) / 360) * 127))
                    
                    for b in buttons:
                        bx, by, bw, bh = b['rect']
                        if bx < px < bx + bw and by < py < by + bh:
                            if b['momentary'] and not b['on']:
                                b['on'] = True; send_midi_note(note=60 + b['id'], state=True)
                            elif not b['momentary'] and b['cd'] == 0:
                                b['on'] = not b['on']; send_midi_note(note=60 + b['id'], state=b['on']); b['cd'] = 25
                else: active_slider_id = None; active_knob_id = None

        final_v = cv2.resize(display, (DISP_W, DISP_H))
        
        # Draw Buttons in Video Stream
        ex, ey, ew, eh = [int(v / SCALE_X if i%2==0 else v / SCALE_Y) for i, v in enumerate(edit_btn_rect)]
        cv2.rectangle(final_v, (ex, ey), (ex + ew, ey + eh), (50, 50, 50), -1)
        cv2.putText(final_v, "EDIT UI", (ex + 40, ey + 50), 1, 1, (255, 255, 255), 2)

        cx, cy, cw, ch = [int(v / SCALE_X if i%2==0 else v / SCALE_Y) for i, v in enumerate(capture_btn_rect)]
        cv2.rectangle(final_v, (cx, cy), (cx + cw, cy + ch), (30, 80, 30), -1)
        cv2.putText(final_v, "CAPTURE PAPER", (cx + 10, cy + 50), 1, 1, (255, 255, 255), 2)

        for b in buttons:
            b['cd'] = max(0, b['cd'] - 1)
            bx, by = int(b['rect'][0] / SCALE_X), int(b['rect'][1] / SCALE_Y)
            bw, bh = int(b['rect'][2] / SCALE_X), int(b['rect'][3] / SCALE_Y)
            if b['on']:
                overlay = final_v.copy()
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), -1)
                final_v = cv2.addWeighted(overlay, 0.4, final_v, 0.6, 0)
            cv2.rectangle(final_v, (bx, by), (bx + bw, by + bh), (0, 255, 0) if b['on'] else (0, 0, 255), 2)

        for s in sliders:
            rx, ry, rw, rh = [int(val / (SCALE_X if i%2==0 else SCALE_Y)) for i, val in enumerate(s['rect'])]
            current_y = ry + int((1.0 - s['val']) * rh)
            cv2.line(final_v, (rx + rw//2, ry + rh), (rx + rw//2, current_y), (0, 255, 255), 4)
            cv2.circle(final_v, (rx + rw//2, current_y), 15, (0, 255, 255), -1)

        for k in knobs:
            kx, ky, kr = int(k['center'][0]/SCALE_X), int(k['center'][1]/SCALE_Y), int(k['radius']/SCALE_X)
            overlay = final_v.copy()
            cv2.ellipse(overlay, (kx, ky), (kr, kr), 0, 0, k['angle'] % 360, (0, 255, 0), -1)
            final_v = cv2.addWeighted(overlay, 0.4, final_v, 0.6, 0)
            cv2.circle(final_v, (kx, ky), kr, (255, 255, 255), 2)
            rad = math.radians(k['angle'])
            cv2.line(final_v, (kx, ky), (int(kx + kr * math.cos(rad)), int(ky + kr * math.sin(rad))), (0, 0, 0), 3)

        cv2.imshow('Hand Interface', final_v)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'ESC' to quit
        if key == 27: 
            break
            
        # Press 'c' to capture paper (ASCII 99)
        elif key == ord('c'):
            cv2.imwrite(INTERFACE_FILE, frame) # Capture the current camera frame
            cap.release()
            cv2.destroyAllWindows()
            return "CAPTURE"

    cap.release()
    cv2.destroyAllWindows()
    return "QUIT"

# --- 4. MAIN FLOW ---
if __name__ == "__main__":
    current_file = run_sketchpad()
    while True:
        status = start_mediapipe_app(current_file)
        if status == "EDIT": 
            current_file = run_sketchpad(existing_file=current_file)
        elif status == "CAPTURE":
            # Paper captured. No need to run sketchpad, just restart the app with the paper photo
            current_file = INTERFACE_FILE
        else: break