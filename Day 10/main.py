import simulation
import pygame
WIDTH, HEIGHT = 800, 600
RED = 255, 0, 0
BLACK = 0, 0, 0
GREEN = 0, 255, 0
BLUE = 0, 0, 255
WHITE = 255, 255, 255
FPS = 120 # 프레임
ALPHA = 1/FPS
INTERVAL = 25 # 격자점 사이 간격

pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation")

N, M, L, K, aK, sK = 16, 1 / 10, 100, 7, 90000, 0.0
DAMP, aDAMP = 0.5, 0.3
mvcN, mvcR = 50, 0.7 * L
HIDEEXTRA = False  # 여기에서 제어점 표시 여부 결정
bonus = [[ 0 for j in range(int(WIDTH/INTERVAL)) ] for i in range(int(HEIGHT/INTERVAL))]

simulation = simulation.Simulation(N, M, L, K, aK, sK, DAMP, aDAMP, bonus, mvcN, mvcR, screen)
do = True
time=0
Action = [ [0,0] for i in range(N) ]
while do:
    fpsClock.tick(FPS)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: do = False
        if event.type == pygame.KEYDOWN: print(simulation.reward())
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            x, y = int(x/INTERVAL+0.5), int(y/INTERVAL+0.5)
            if event.button == 1: bonus[y][x] = 1
            elif event.button == 3: bonus[y][x] = -1

    for i in range(int(N/2)):
        if time<150: continue
        Action[i] = [-60,60] if time<350 else [0,0]
        Action[(int(N/2)+i)%N] = [0,0] if time<400 else[60,-60]

    simulation.step(Action)
    simulation.render(HIDEEXTRA)
    pygame.display.flip()

for i in range(int(HEIGHT/INTERVAL)):
    string = ""
    for j in range (int(WIDTH/INTERVAL)):
        string += str(bonus[i][j])+" "
    print(string)