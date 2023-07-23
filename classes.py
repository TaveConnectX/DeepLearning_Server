import pygame

SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
PURPLE = (255,0,255)
BLACK = (0,0,0)
WHITE = (255,255,255)
LIGHTGRAY = (180,180,180)
DARKGRAY = (127,127,127)

P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

class Button:
    def __init__(self, name, cx=0,cy=0,width=100,height=100,color=WHITE, font=None,font_size=30):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.color = color
        self.font = pygame.font.SysFont(font, font_size, bold=False, italic=False)
        self.clicked = False
        self.clicked_color = LIGHTGRAY

    def draw_and_get_event(self, surface, event):
        x = self.cx-self.width/2
        y = self.cy-self.height/2
        # print(self.x,self.y, self.width, self.height)
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        # print(click)
        border = pygame.draw.rect(surface,self.color, (x,y,self.width,self.height))
        if x+self.width > mouse_pos[0] > x and y+self.height > mouse_pos[1] > y:
            

            if event is not None:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.clicked_color = DARKGRAY
                    self.clicked = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.clicked: 
                        self.clicked = False
                        self.clicked_color = LIGHTGRAY
                        return True
                    else:
                        self.clicked = False

            pygame.draw.rect(surface, self.clicked_color,(x,y,self.width,self.height))
            # if click[0] == 1:
            #     print(click)     
            #     return True
        elif self.clicked:
            if event.type == pygame.MOUSEBUTTONUP:
                self.clicked = False
                self.clicked_color = LIGHTGRAY
            pygame.draw.rect(surface, self.clicked_color,(x,y,self.width,self.height))
        else:
            border = pygame.draw.rect(surface,self.color, (x,y,self.width,self.height))
        text = self.font.render(self.name, True, BLACK)
        text_rect = text.get_rect(center=(surface.get_width()/2, surface.get_height()/2))
        text_rect.center = border.center
        surface.blit(text, text_rect)

        return False