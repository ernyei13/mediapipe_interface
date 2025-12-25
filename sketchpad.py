import os
import pygame

# Configuration matching the main app
RES_W, RES_H = 1920, 1080
DISP_W, DISP_H = 1280, 720
SCALE_X = RES_W / DISP_W
SCALE_Y = RES_H / DISP_H
INTERFACE_FILE = "custom_interface.png"
PRESETS_DIR = "presets"

def run_sketchpad(existing_file=None):
    pygame.init()
    screen = pygame.display.set_mode((DISP_W, DISP_H))
    pygame.display.set_caption("Sketchpad - Draw Interface")
    
    canvas = pygame.Surface((RES_W, RES_H))
    canvas.fill((255, 255, 255))
    undo_stack = [] 

    def get_presets(): 
        if not os.path.exists(PRESETS_DIR): os.makedirs(PRESETS_DIR)
        return [f for f in os.listdir(PRESETS_DIR) if f.endswith(".png")]
    
    preset_files = get_presets()

    if existing_file and os.path.exists(existing_file):
        try:
            loaded_img = pygame.image.load(existing_file)
            canvas.blit(pygame.transform.scale(loaded_img, (RES_W, RES_H)), (0, 0))
        except: pass
    
    undo_stack.append(canvas.copy())
    font = pygame.font.SysFont("Arial", 18, bold=True)
    drawing, running, last_pos, mode = False, True, None, "PEN" 
    dropdown_open, saving_preset, preset_name = False, False, ""

    while running:
        screen.fill((200, 200, 200))
        keys = pygame.key.get_pressed()
        ctrl_cmd = keys[pygame.K_LCTRL] or keys[pygame.K_LMETA] 

        btn_try = pygame.Rect(DISP_W - 140, DISP_H - 50, 120, 40)
        btn_undo = pygame.Rect(10, DISP_H - 50, 70, 40)
        btn_clear = pygame.Rect(90, DISP_H - 50, 70, 40)
        btn_presets = pygame.Rect(170, DISP_H - 50, 100, 40)
        btn_save_pre = pygame.Rect(280, DISP_H - 50, 110, 40)
        btn_pen = pygame.Rect(400, DISP_H - 50, 60, 40)
        btn_erase = pygame.Rect(470, DISP_H - 50, 80, 40)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            
            if event.type == pygame.KEYDOWN and ctrl_cmd:
                if event.key == pygame.K_z and len(undo_stack) > 1:
                    undo_stack.pop(); canvas = undo_stack[-1].copy()
                if event.key == pygame.K_k:
                    undo_stack.append(canvas.copy()); canvas.fill((255, 255, 255))

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
                if dropdown_open:
                    for i, file in enumerate(preset_files):
                        item_rect = pygame.Rect(170, DISP_H - 95 - (i * 30), 100, 30)
                        if item_rect.collidepoint(event.pos):
                            undo_stack.append(canvas.copy())
                            img = pygame.image.load(os.path.join(PRESETS_DIR, file))
                            canvas.blit(pygame.transform.scale(img, (RES_W, RES_H)), (0, 0))
                            dropdown_open = False; break
                
                if btn_try.collidepoint(event.pos): pygame.image.save(canvas, INTERFACE_FILE); running = False 
                elif btn_presets.collidepoint(event.pos): dropdown_open = not dropdown_open
                elif btn_save_pre.collidepoint(event.pos): saving_preset = True
                elif btn_undo.collidepoint(event.pos):
                    if len(undo_stack) > 1: undo_stack.pop(); canvas = undo_stack[-1].copy() 
                elif btn_clear.collidepoint(event.pos):
                    undo_stack.append(canvas.copy()); canvas.fill((255, 255, 255))
                elif btn_pen.collidepoint(event.pos): mode = "PEN"
                elif btn_erase.collidepoint(event.pos): mode = "ERASER"
                elif event.pos[1] < DISP_H - 60 and not dropdown_open:
                    undo_stack.append(canvas.copy()); drawing, last_pos = True, canvas_pos

            if event.type == pygame.MOUSEBUTTONUP: drawing = False
            if event.type == pygame.MOUSEMOTION and drawing:
                color, thick = ((0, 0, 0), 8) if mode == "PEN" else ((255, 255, 255), 60)
                if last_pos: pygame.draw.line(canvas, color, last_pos, canvas_pos, thick)
                last_pos = canvas_pos

        screen.blit(pygame.transform.smoothscale(canvas, (DISP_W, DISP_H)), (0, 0))
        pygame.draw.rect(screen, (40, 40, 180), btn_try); screen.blit(font.render("TRY", True, (255,255,255)), (btn_try.x+40, btn_try.y+11))
        pygame.draw.rect(screen, (50, 180, 50) if mode == "PEN" else (100, 100, 100), btn_pen); screen.blit(font.render("PEN", True, (255,255,255)), (btn_pen.x+12, btn_pen.y+11))
        pygame.draw.rect(screen, (50, 180, 50) if mode == "ERASER" else (100, 100, 100), btn_erase); screen.blit(font.render("RUBBER", True, (255,255,255)), (btn_erase.x+5, btn_erase.y+11))
        pygame.display.flip()
    pygame.quit(); return INTERFACE_FILE
