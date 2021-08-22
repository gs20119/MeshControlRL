
import pygame
import math

WIDTH, HEIGHT = 800, 600
RED = 255, 0, 0
BLACK = 0, 0, 0
GREY = 75, 75, 75
GREEN = 0, 255, 0
BLUE = 0, 0, 255
WHITE = 255, 255, 255
FPS = 120 # 프레임
ALPHA = 1/FPS

N, aK, bK = 36, 3, 1
INTERVAL = 20
DV, TV, RV = 5, 5, math.pi/11  # Deformate, Rotate, Translate
bonus = [[ 0 for j in range(int(HEIGHT/INTERVAL)) ] for i in range(int(WIDTH/INTERVAL))]

pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation")


class Simulation:
    def __init__(self):
        self.Dir = 0.501*math.pi
        self.Ctrl = [75 for i in range(N)]
        self.Pos = [WIDTH/2,HEIGHT/2]

    def getCtrl(self, i):
        x, y = self.Pos[0], self.Pos[1]
        x += self.Ctrl[i]*math.cos(self.Dir+2*math.pi*i/N)
        y += self.Ctrl[i]*math.sin(self.Dir+2*math.pi*i/N)
        return x, y
        
    def draw(self, screen):
        for i in range(N):
            pygame.draw.circle(screen, BLACK, self.getCtrl(i), 5)
        for i in range(int(WIDTH/INTERVAL)):
            for j in range(int(HEIGHT/INTERVAL)):
                x, y = i*INTERVAL, j*INTERVAL
                if bonus[i][j] == 0:
                    col = GREEN if self.isInside(x,y) else GREY
                else: col = BLUE if bonus[i][j]<0 else RED
                pygame.draw.circle(screen, col, (x,y), 3)

    def Deformate(self, I, move):
        K = aK if move>0 else bK
        for i in range(I-K, I+K+1):
            k = (i+N)%N
            self.Ctrl[k] += move*DV
            if self.Ctrl[k]<0 : self.Ctrl[k]=0

    def Translate(self, moveX, moveY):
        self.Pos[0] += moveX*TV
        self.Pos[1] += moveY*TV

    def Rotate(self, move):
        self.Dir += move*RV
        if self.Dir > 2*math.pi : self.Dir -= 2*math.pi

    def reward(self):
        Reward = 0
        for i in range(int(WIDTH/INTERVAL)):
            for j in range(int(HEIGHT/INTERVAL)):
                x, y = i*INTERVAL, j*INTERVAL
                if self.isInside(x,y): Reward += bonus[i][j]
        return Reward

    def isInside(self, x, y):
        cross = 0
        for i in range(N):
            Pos1, Pos2 = self.getCtrl(i), self.getCtrl((i+1)%N)
            if (Pos1[1]>y and Pos2[1]<y) or (Pos1[1]<y and Pos2[1]>y):
                atX = (y-Pos2[1])*(Pos1[0]-Pos2[0])/(Pos1[1]-Pos2[1]) + Pos2[0]
                if x<atX: cross += 1
        return cross % 2 > 0


Sim = Simulation()

while True:
    fpsClock.tick(FPS)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: exit()
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: Sim.Rotate(1)
            if keys[pygame.K_RIGHT]: Sim.Rotate(-1)
            if keys[pygame.K_a]: Sim.Translate(-1,0)
            if keys[pygame.K_d]: Sim.Translate(1,0)
            if keys[pygame.K_s]: Sim.Translate(0,1)
            if keys[pygame.K_w]: Sim.Translate(0,-1)

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            x -= Sim.Pos[0]
            y -= Sim.Pos[1]
            theta = math.pi-math.atan2(y,-x)-Sim.Dir
            k = int(theta/(2*math.pi/N))
            if event.button == 1: Sim.Deformate(k,1)
            if event.button == 3: Sim.Deformate(k,-1)
            
    screen.fill(WHITE)
    Sim.draw(screen)
    pygame.display.update()

