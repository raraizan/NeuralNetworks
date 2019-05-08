import pygame
from pygame.locals import *
 
class PrimitiveSketch(object):
    """ Intended to deal with the basic initialization
    and basic sketch variables like width, height
    mouse position and time handling.
    Deals with the main loop and event handling.
    """

    def __init__(self):

        pygame.init()
        self._running = True
        self.fps = 60
        self.setup()
        self.clock = pygame.time.Clock()
        
        self._display_surf = pygame.display.set_mode((self.width, self.height))

        while self._running:
            # update frame

            self.clock.tick(self.fps)
            self.frame_rate = self.clock.get_fps()

            # Get events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            self.mouseX, self.mouseY = pygame.mouse.get_pos()
            self.mousePressed = any(pygame.mouse.get_pressed())
            # Draw stuff
            self.draw()
            pygame.display.flip()

        self.cleanup()
 
    def setup(self):
        pass

    def draw(self):
        pass

    def size(self, width, height):
        self.width, self.height = width, height

    def set_frame_rate(self, fps):
            self.fps = fps

    def cleanup(self):
        pygame.quit()

class Sketch(PrimitiveSketch):
    
    def __init__(self):
        self._actual_fill_color = (255, 255, 255)
        self._actual_stroke_color = (0, 0, 0)
        self._stroke = True
        self._fill = True
        self._actual_stroke_weight = 1
        self._rect_offset_mult = 0.0 # default: corner

        super().__init__()

    # TODO: implement use of alpha and 1 argument for grayscale, two
    # for grayscale and alpha, 3 for rgb and 4 for rgb and alpha

    # Color Handling

    def background(self, color):
        self._display_surf.fill(color)

    def fill(self, color):
        self._fill = True
        self._actual_fill_color = color
    
    def stroke(self, color):
        self._stroke = True
        self._actual_stroke_color = color
    
    def stroke_weight(self, sw):
        self._actual_stroke_weight = sw

    # def _get_grayscale(self, value)
    #     if 0 <= value < 256:
    #         return (value, value, value)
    #     else:
    #         # raise
    #         print()
    def no_stroke(self):
        self._stroke = False
    
    def no_fill(self):
        self._fill = False

    def line(self, x1, y1, x2, y2):
        if self._stroke:
            pygame.draw.aaline(
                self._display_surf,
                self._actual_stroke_color,
                (x1, y1),
                (x2, y2),
                self._actual_stroke_weight,
            )

    def rect(self, x, y, w, h):

        if self._fill:
            pygame.draw.rect(
                self._display_surf,
                self._actual_fill_color,
                (
                    x - self._rect_offset_mult * h,
                    y - self._rect_offset_mult * w,
                    w,
                    h,
                ),
                0,
            )
        if self._stroke:
            pygame.draw.rect(
                self._display_surf,
                self._actual_stroke_color,
                (
                    x - self._rect_offset_mult * h,
                    y - self._rect_offset_mult * w,
                    w,
                    h,
                ),
                self._actual_stroke_weight,
            )

    def ellipse(self, x, y, w, h):
        if self._fill:
            pygame.draw.ellipse(
                self._display_surf,
                self._actual_fill_color,
                (
                    x - self._rect_offset_mult * h,
                    y - self._rect_offset_mult * w,
                    w,
                    h,
                ),
                0,
            )
        if self._stroke:
            pygame.draw.ellipse(
                self._display_surf,
                self._actual_stroke_color,
                (
                    x - self._rect_offset_mult * h,
                    y - self._rect_offset_mult * w,
                    w,
                    h,
                ),
                self._actual_stroke_weight,
            )

    def rect_mode(self, mode):

        modes = {
            'corner': 0.0,
            'center': 0.5,
        }

        if mode in modes:
            self._rect_offset_mult = modes[mode]
        else:
            # raise invalid mode
            print("Invalid or not implemented mode")
