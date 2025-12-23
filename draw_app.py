import pygame
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CANVAS_HEIGHT = 520  # Space for the drawing
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
BUTTON_COLOR = (50, 50, 50)

# Setup Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Black Pen Sketchpad")

# Create a separate surface for drawing (this is what gets saved)
canvas = pygame.Surface((WIDTH, CANVAS_HEIGHT))
canvas.fill(WHITE)

# Font for button
font = pygame.font.SysFont("Arial", 20, bold=True)

def get_next_filename():
    """Checks the folder and returns 'interface_N.png' with the next available number."""
    counter = 1
    while os.path.exists(f"interface_{counter}.png"):
        counter += 1
    return f"interface_{counter}.png"

def main():
    drawing = False
    last_pos = None
    running = True

    while running:
        # 1. Fill the background of the UI area
        screen.fill(GRAY)
        
        # 2. Display the drawing canvas on the screen
        screen.blit(canvas, (0, 0))

        # 3. Create Save Button Rect
        button_rect = pygame.Rect(WIDTH // 2 - 60, CANVAS_HEIGHT + 15, 120, 40)
        pygame.draw.rect(screen, BUTTON_COLOR, button_rect, border_radius=5)
        text_surf = font.render("SAVE", True, WHITE)
        screen.blit(text_surf, (button_rect.x + 35, button_rect.y + 8))

        # 4. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check if we clicked the SAVE button
                if button_rect.collidepoint(event.pos):
                    filename = get_next_filename()
                    pygame.image.save(canvas, filename)
                    print(f"Image saved as: {filename}")
                # Otherwise, start drawing if within canvas bounds
                elif event.pos[1] < CANVAS_HEIGHT:
                    drawing = True
                    last_pos = event.pos

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None

            if event.type == pygame.MOUSEMOTION:
                if drawing:
                    current_pos = event.pos
                    # Only draw if the mouse is still within the drawing area
                    if current_pos[1] < CANVAS_HEIGHT:
                        if last_pos:
                            # Draw a line from the last mouse position to the current one
                            pygame.draw.line(canvas, BLACK, last_pos, current_pos, 3)
                        last_pos = current_pos
                    else:
                        # Stop drawing if the mouse leaves the canvas area
                        last_pos = None

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()