import pygame
import sys
import numpy as np
from classes import *
from test_model import test_main
pygame.init()


SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Clock 객체 생성
clock = pygame.time.Clock()

# pygame.Rect(x,y,width, height)
# myRect = pygame.Rect(150, 200, 200, 100)




            

def intro():
    w, h = SCREEN.get_size()
    intro_button_width = w//3
    intro_button_height = h//16
    print(w,h,intro_button_width,intro_button_height)
    intro_buttons = [
        Button('new game',cx=w/2,cy=h/2+intro_button_height*1,width=intro_button_width, height=intro_button_height),
        Button('continue',cx=w/2,cy=h/2+intro_button_height*2,width=intro_button_width, height=intro_button_height),
        Button('how to',cx=w/2,cy=h/2+intro_button_height*3,width=intro_button_width, height=intro_button_height),
        Button('review',cx=w/2,cy=h/2+intro_button_height*4,width=intro_button_width, height=intro_button_height),
        Button('option',cx=w/2,cy=h/2+intro_button_height*5,width=intro_button_width, height=intro_button_height),
        Button('quit',cx=w/2,cy=h/2+intro_button_height*6,width=intro_button_width, height=intro_button_height)
    ]
    logo_image = pygame.image.load('image/logo.png')
    logo_image = pygame.transform.scale(logo_image, (w/1.1, w/1.1/4))
    logo_rect = logo_image.get_rect()
    logo_rect.x = w/2- logo_image.get_width()/2
    logo_rect.y = h/4- logo_image.get_height()/2
    
    run = True
    SCREEN.fill((255,255,255))
    while run:
        SCREEN.blit(logo_image, logo_rect)
        for event in pygame.event.get():
            
            for button in intro_buttons:
                action = button.draw_and_get_event(SCREEN,event)
                if action:
                    print(button.name)
                    if button.name == 'new game': select_difficulty()
                    elif button.name == 'continue': play(cont_game=True)
                    elif button.name == 'how to': how_to()
                    elif button.name == 'review': review()
                    elif button.name == 'option': option()
                    elif button.name == 'quit': run=False
                    else:
                        print('error')
            if event.type == pygame.QUIT:
                run = False
        clock.tick(60)
        pygame.display.flip()


    pygame.quit()
    sys.exit()

def draw_table():
    w,h = SCREEN.get_size()
    block_size = (w-100)/7
    for i in range(8):
        pygame.draw.line(SCREEN, BLACK, (50+block_size*i,100), (50+block_size*i,100+6*block_size), width=5)
    for i in range(7):
        pygame.draw.line(SCREEN, BLACK, (50,100+block_size*i), (50+block_size*7,100+block_size*i), width=5)


def draw_cursor(x, player):
    if player == 1: color = P1COLOR
    else: color = P2COLOR
    pygame.draw.polygon(SCREEN, color, ((x,90),(x-10,80),(x+10,80)), width=0)

def draw_circle_with_pos(pos,player):
    w,h = SCREEN.get_size()
    r = (w-100)/7/2/1.05
    if player == 1: color = P1COLOR
    else: color = P2COLOR
    x,y = pos
    x += 0.5
    y += 0.5
    pos = (x,y)
    pygame.draw.circle(SCREEN,color,pos,r)


# board 상의 좌표를 SCREEN의 좌표로 변경
def cord2pos(cord):
    y,x = cord
    # y = 5-y
    w,h = SCREEN.get_size()
    r = (w-100)/7/2

    # (0,0) 일 때의 position
    pos = [50+r,100+r]
    pos[0] += 2*x*r
    pos[1] += 2*y*r

    return pos

def x2col(x):
    w,h = SCREEN.get_size()
    block_size = (w-100)/7
    x -= 50
    col = -1

    while x >= 0:
        x -= block_size
        col += 1

    if not 0<=col<=6:
        col = -1
    return col

def is_valid_x(x):
    w,h = SCREEN.get_size()
    if x<50 or w-50<x:
        return False
    else: return True

def get_next_state(board, col,player):
    if col == -1: 
        return board, player
    if board[0][col] != 0:
        return board, player
    
    for row in range(5,-1,-1):
        if board[row][col] == 0:
            board[row][col] = player
            return board, 2//player

# made by chatgpt and I edit little bit.
# 가로, 세로, 대각선에 완성된 줄이 있는지를 체크한다 
def is_win(board, player):
    for i in range(6):
        for j in range(7):
            if board[i][j] == player:
                # horizontal
                if j + 3 < 7 and board[i][j+1] == board[i][j+2] == board[i][j+3] == player:
                    return player
                # vertical
                if i + 3 < 6 and board[i+1][j] == board[i+2][j] == board[i+3][j] == player:
                    return player
                # diagonal (down right)
                if i + 3 < 6 and j + 3 < 7 and board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == player:
                    return player
                # diagonal (up right)
                if i - 3 >= 0 and j + 3 < 7 and board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == player:
                    return player
                

    if 0 not in board[0,:]: player = 3
    return 0

def select_difficulty():
    w, h = SCREEN.get_size()
    # 버튼 3개 만들고
    easy_button = Button('easy',cx=w/2,cy=h/2/2,width=w/3)
    normal_button = Button('normal',cx=w/2,cy=h/2,width=w/3)
    hard_button = Button('hard',cx=w/2,cy=h/4*3,width=w/3)

    action = False
    # 선택하면 play()    
    run = True
    event = None
    SCREEN.fill((255,255,255))
    easy_action = easy_button.draw_and_get_event(SCREEN,event)
    normal_action = normal_button.draw_and_get_event(SCREEN,event)
    hard_action = hard_button.draw_and_get_event(SCREEN,event)
    while run:
        for event in pygame.event.get():
            
            

            easy_action = easy_button.draw_and_get_event(SCREEN,event)
            normal_action = normal_button.draw_and_get_event(SCREEN,event)
            hard_action = hard_button.draw_and_get_event(SCREEN,event)
            if event.type == pygame.QUIT:
                run = False
        if easy_action:
            play(difficulty='easy')
            return
        elif normal_action:
            play(difficulty='normal')
            return
        elif hard_action:
            play(difficulty='hard')
            return
        clock.tick(60)
        pygame.display.flip()

def play(difficulty,cont_game=False):
    print('play')
    player = np.random.choice([1,2])
    back_button = Button('<-')
    go_back = False
    if cont_game:
        '''
        여기엔 이어할 보드를 불러오는 기능이 필요함
        '''
        board = np.zeros((6,7))
    else:
        board = np.zeros((6,7))
        # board = [[np.random.choice([1, 2]) for _ in range(7)] for _ in range(6)]
        print(board)
    run = True
    event = None
    x, y = 50,100
    clicked_x, clicked_y = 0,0
    SCREEN.fill(WHITE)
    draw_table()
    draw_cursor(x,player)
    pygame.display.flip()
    while run:
        SCREEN.fill(WHITE)
        if is_win(board,2//player) != 0:
            end(board, is_win(board,2//player))
            return
        if player == 2:
            col = test_main(board, difficulty)
            board, player = get_next_state(board,col,player)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_x, clicked_y = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONUP:
                print(clicked_x)
                if not is_valid_x(clicked_x): continue
                x,_ = pygame.mouse.get_pos()
                col = x2col(x)
                board, player = get_next_state(board,col,player)
                print(board)

            x,y = pygame.mouse.get_pos()
            
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
        go_back = back_button.draw_and_get_event(SCREEN, event)
        if go_back: 
            SCREEN.fill(WHITE)
            return
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        draw_table()
        draw_cursor(x,player)
        pygame.display.flip()

def end(board, player):
    w,h = SCREEN.get_size()
    if player == 1: text_content = "이겼습니다! 축하드립니다!"
    elif player == 2: text_content = "아쉽게도 졌네요 ㅠㅠ"
    else: text_content = "비겼습니다! 한번 더하면 이길지도...?"

    back_button = Button('<-',cx=w/2,cy=h*3/4,width=w/3,height=100)

    border = pygame.draw.rect(SCREEN, WHITE, (0,h/1.75,w,100))
    font = pygame.font.SysFont('malgungothic', 30)
    text = font.render(text_content, True, BLACK)
    text_rect = text.get_rect(center=(SCREEN.get_width()/2, SCREEN.get_height()/2))
    text_rect.center = border.center

    run = True
    event = None
    go_back = False
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SCREEN.fill(WHITE)
                run = False
        if go_back: 
            SCREEN.fill(WHITE)
            return
        go_back = back_button.draw_and_get_event(SCREEN, event)
        SCREEN.blit(text, text_rect)
        pygame.display.flip()

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    pos = cord2pos((i,j))
                    draw_circle_with_pos(pos, player=board[i][j])
        draw_table()
        

def how_to():
    pass
def review():
    pass
def option():
    pass
# 이미지 로드
# pygame.image.load(image_file)
# pygame.transform.scale(image object, (width, height)) 

# SCREEN.fill((R,G,B))
# SCREEN.fill( (255, 255, 255) )

# 이미지를 화면에 출력 
# SCREEN.blit(image object,  cord or Rect)

# 선 그리기
# pygame.draw.line(SCREEN, color, start_pos, end_pos, width=1) -> Rect
# 선 여러개 그리기
# pygame.draw.lines(SCREEN, color, closed, points, width=1) -> Rect
# 부드러운 선
# pygame.draw.aaline(SCREEN, color, start_pos, end_pos, blend=1) -> Rect
# 부드러운 선 여러개 그리기
# pygame.draw.aaline(SCREEN, color, closed, points, blend=1) -> Rect
# 네모 그리기
# pygame.draw.rect(SCREEN, color, rect, width=0) -> Rect
# 다각형 그리기
# pygame.draw.polygon(surface, color, points, width=0) -> Rect
# 원 그리기
# pygame.draw.circle(surface, color, center, radius, width=0) -> Rect
# 타원 그리기
# pygame.draw.ellipse(surface, color, rect, width=0) -> Rect

# 충돌 체크 방법
'''
def collision_check(A, B):
    if A.top < B.bottom and B.top < A.bottom and A.left < B.right and B.left < A.right:
        return True
    else:
        return False

A_rect = A_object.get_rect()
B_rect  = B_object.get_rect()

collision_check(A_rect, B_rect)
'''

# 사각형을 이미지로 이용하기 위해서
# rect_image = pygame.Surface((10,20))
# rect_image.fill(RED)
# 이미지의 Rect 정보를 저장

# player_Rect = rect_image.get_rect()


# print( pygame.font.get_fonts() )
# 폰트 설정
# my_font = pygame.font.SysFont('arial', 30, bold=False, italic=False)

# render(text,antialias, color, background)
# text_Title = my_font.render("Pygame Text Test", True, BLACK)
# 이미지가 가운데 올 수 있도록 좌표값 수정
# python 3.8 이상에서 integer가 필요한 곳에 float가 들어가면 DeprecationWarning이 나옴.
# 따라서 round() 처리를 해준다.
# player_Rect.centerx = round(SCREEN_WIDTH / 2)
# player_Rect.centery = round(SCREEN_HEIGHT / 2)

dx, dy = 0, 0
# 화면 업데이트
# pygame.display.update()
# pygame.display.filp()
pygame.display.set_caption('Connect 4 by Team Connect X of TAVE 11th')
def main():
    intro()
    

if __name__ == '__main__':
    main()
