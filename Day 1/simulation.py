
import pygame
import math

WIDTH, HEIGHT = 800, 600
RED = 255, 0, 0
BLACK = 0, 0, 0
WHITE = 255, 255, 255
FPS = 60 # 프레임
ALPHA = 1/FPS
FORCE = 100 # 키보드로 입력하는 힘
K = 10 # 탄성계수
DAMP = 0.7
COUNT = 8 # 점들의 개수

adj = [ # 인접행렬, 숫자는 자연길이
    [0, 50, 0, 0, 0, 0, 50, 60],
    [50, 0, 50, 0, 0, 0, 0, 60],
    [0, 50, 0, 50, 0, 0, 0, 60],
    [0, 0, 50, 0, 50, 0, 0, 60],
    [0, 0, 0, 50, 0, 50, 0, 60],
    [0, 0, 0, 0, 50, 0, 50, 60],
    [50, 0, 0, 0, 0, 50, 0, 60],
    [60, 60, 60, 60, 60, 60, 60, 0]
]

defPos = [ # 초기 정점들의 위치
    [28,194],
    [-27,138],
    [-24,44],
    [60,15],
    [154,49],
    [157,151],
    [90,194],
    [60,102]
] 

pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation")


class Particle:
    def __init__(self, pos, m=1):
        self.pos = pos
        self.m = m
        self.m, self.v, self.a = 1, [0,0], [0,0]
    def draw(self, screen):
        pygame.draw.circle(screen, RED, self.pos, 5)
    def drawEdge(self, P, screen):
        pygame.draw.line(screen, BLACK, self.pos, P.pos, 5)

    def update(self):
        self.v[0], self.v[1] = self.v[0]+self.a[0]*ALPHA, self.v[1]+self.a[1]*ALPHA
        self.pos[0], self.pos[1] = self.pos[0]+self.v[0]*ALPHA, self.pos[1]+self.v[1]*ALPHA

    def delta(self, P):
        return P.pos[0]-self.pos[0], P.pos[1]-self.pos[1]
    def dist(self, P):
        x, y = self.delta(P)
        return math.sqrt(x**2+y**2)
    def dir(self, P):
        x, y = self.delta(P)
        theta = math.atan(y/x) if x!=0 else math.pi/2
        i, j = math.cos(theta), math.sin(theta)
        if x*i<0 or y*j<0: i, j = -i, -j
        return [i,j]


class Simulation:
    def __init__(self):
        self.Vtx = [ Particle(defPos[i]) for i in range(COUNT) ]
    
    def draw(self, screen):
        for i in range(COUNT):
            for j in range(i,COUNT):
                if adj[i][j]!=0: self.Vtx[i].drawEdge(self.Vtx[j],screen)
            self.Vtx[i].draw(screen)
    
    def calculate(self, Ext):
        for i in range(COUNT):
            self.Vtx[i].a = [0,0]
            for j in range(COUNT):
                if adj[i][j]!=0:
                    dir = self.Vtx[i].dir(self.Vtx[j])
                    pull = self.Vtx[i].dist(self.Vtx[j]) - adj[i][j]
                    self.Vtx[i].a[0] += (pull*dir[0]*K+Ext[i][0])/self.Vtx[i].m - DAMP*self.Vtx[i].v[0]
                    self.Vtx[i].a[1] += (pull*dir[1]*K+Ext[i][1])/self.Vtx[i].m - DAMP*self.Vtx[i].v[1]
        for i in range(COUNT):
            self.Vtx[i].update()


Sim = Simulation()

while True:
    fpsClock.tick(FPS)
    External = [[0,0] for i in range(COUNT)]
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]: External[0][1] -= FORCE
    if keys[pygame.K_DOWN]: External[0][1] += FORCE
    if keys[pygame.K_RIGHT]: External[0][0] += FORCE
    if keys[pygame.K_LEFT]: External[0][0] -= FORCE
    if keys[pygame.K_w]: External[2][1] -= FORCE
    if keys[pygame.K_a]: External[2][0] -= FORCE
    if keys[pygame.K_s]: External[2][1] += FORCE
    if keys[pygame.K_d]: External[2][0] += FORCE

    print(External)
    screen.fill(WHITE)
    Sim.calculate(External)
    Sim.draw(screen)
    pygame.display.update()

