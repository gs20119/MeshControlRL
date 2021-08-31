import math
import pygame
import functions as f

#이용할 색깔
BLACK = 0, 0, 0
WHITE = 255, 255, 255
BLUE = 0, 0, 255
GREEN = 0, 255, 0
RED = 255, 0, 0
GREY = 75, 75, 75

WIDTH, HEIGHT = 800, 600
FPS = 120
ALPHA = 1 / FPS
r = 5
LineWidth = 5
INTERVAL = 25
Size = [WIDTH, HEIGHT]


class Physics: # 가속도, 각가속도가 여기에서 처리됨
    def __init__(self, value):
        self.x = value
        self.v = 0
        self.a = 0

    def update(self):
        self.v += self.a * ALPHA
        self.x += self.v * ALPHA
        self.a = 0

    def accel(self, value):
        self.a += value

    def getPos(self):
        return self.x

    def getSpeed(self):
        return self.v


class MVCCurve: # Mean Value Coordinate 곡선을 처리
    def __init__(self, Obj, N, Radius): 
        self.R0, self.N = Radius, N
        self.Weight = [[self.setWeight(Obj,i,j) for j in range(Obj.N)] for i in range(self.N)]
        self.Curve = [(0, 0) for i in range(self.N)]

    def setWeight(self, Obj, i, j): # 초기 가중치 설정
        point = f.add(Obj.getCenter(), f.mul(mvcR, f.unitCircle(2*math.pi*i/self.N)))
        prev = j-1 if j>0 else Obj.N-1
        next = (j+1)%Obj.N
        ret = math.tan(abs(f.angle(Obj.getControl(prev), Obj.getControl(j), point))/2)
        ret += math.tan(abs(f.angle(Obj.getControl(next), Obj.getControl(j), point))/2)
        ret /= f.dist(Obj.getControl(j), point)
        return ret

    def update(self, Obj): # Obj를 따라 Curve 업데이트하기
        for i in range(self.N):
            fx, fy, totalW = 0, 0, 0
            for j in range(Obj.N):
                totalW += self.Weight[i][j]
                fx += Obj.getControl(j)[0]*self.Weight[i][j]
                fy += Obj.getControl(j)[1]*self.Weight[i][j]
            self.Curve[i] = fx/totalW, fy/totalW

    def draw(self): # Curve 그리기
        for i in range(self.N):
            x, y = self.Curve[i]
            pygame.draw.circle(screen, BLACK, (x,y), 2*r/3)

    def isInside(self, x, y): # 어떤 점이 Curve 내부에 있는지 판정
        cross = 0
        for i in range(self.N):
            Pos1, Pos2 = self.Curve[i], self.Curve[(i+1)%self.N]
            if (Pos1[1] > y and Pos2[1] <= y) or (Pos1[1] <= y and Pos2[1] > y):
                if Pos1[1] == Pos2[1]:
                    if x<=max(Pos1[0], Pos2[0]): cross += 1
                else:
                    atX = (y-Pos2[1])*(Pos1[0]-Pos2[0])/(Pos1[1]-Pos2[1]) + Pos2[0]
                    if x <= atX: cross += 1
        return cross % 2 > 0



class Object:
    def __init__(self, N, Mass, Radius, k, aK, damp, aDamp, mvcN, mvcR):
        self.N, self.mass, self.k, self.aK, self.R0, self.damp, self.aDamp = (
            N, Mass, k, aK, Radius, damp, aDamp)
        self.Control = [(Physics(self.R0*math.cos(2*math.pi*i/self.N)+WIDTH/2), # 제어점 (물리현상 근사)
                          Physics(self.R0*math.sin(2*math.pi*i/self.N)+HEIGHT/2)) for i in range(self.N)]
        self.Center = [Physics(WIDTH/2), Physics(HEIGHT/2)] # 오브젝트의 중심점
        self.mainAxis = Physics(0) # 오브젝트의 주축
        self.totalI = (self.N*(self.R0**2)*self.mass) # 관성모멘트
        self.Tqnet = 0 # 알짜 토크
        self.Fnet = list([(0, 0) for i in range(self.N+1)]) # 알짜힘
        for i in range(self.N+1): self.Fnet[i] = list(self.Fnet[i]) 
        self.Curve = MVCCurve(self,mvcN,mvcR) # MVC 곡선 (관찰의 대상)

    def reset(self): # 오브젝트 초기화
        self.Control = [(Physics((1)*self.R0*math.cos(2*math.pi*i/self.N) + WIDTH/2),
                          Physics((1)*self.R0*math.sin(2*math.pi*i/self.N) + HEIGHT/2)) for i in range(self.N)]
        self.Center = [Physics(WIDTH/2), Physics(HEIGHT/2)]
        self.mainAxis = Physics(0)

    def extForce(self, F): # 외력 추가하기 (1)
        for i in range(self.N):
            self.Fnet[i][0] += F[i][0]
            self.Fnet[i][1] += F[i][1]

    def calculate(self): # 탄성에 의한 힘과 토크 계산하기 (2) 
        Center = self.getCenter()
        for i in range(self.N):
            pos = self.getControl(i)
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(self.k * (self.R0-f.dist(pos,Center)), f.dir(Center,pos)))
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(-self.damp, self.getVelocity(i)))
            self.Fnet[self.N] = f.add(self.Fnet[self.N], f.mul(-self.k * (self.R0-f.dist(pos,Center)), f.dir(Center,pos)))
            theta = math.atan2(pos[1]-self.getCenter()[1], pos[0]-self.getCenter()[0])
            Tq = self.aK * f.setArrange(theta-(2*math.pi*i/self.N)-self.mainAxis.getPos(), -math.pi, math.pi, 2*math.pi)
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(Tq / (self.mass*(f.dist(pos, Center)**2)) , f.unitCircle(theta-1*math.pi/2)))
            self.totalI += self.mass * (f.dist(pos,Center)**2)
            self.Tqnet += Tq

    def update(self): # 상태 및 결과 업데이트하기 (3)
        for i in range(self.N):
            self.Control[i][0].accel(self.Fnet[i][0]/self.mass)
            self.Control[i][1].accel(self.Fnet[i][1]/self.mass)
        self.Center[0].accel(self.Fnet[self.N][0]/(self.N * self.mass))
        self.Center[1].accel(self.Fnet[self.N][1]/(self.N * self.mass))
        self.mainAxis.accel(self.Tqnet / self.totalI)
        for i in range(self.N+1): self.Fnet[i][0] = self.Fnet[i][1] = 0
        self.totalI = self.Tqnet = 0

        for i in range(self.N):
            for val in self.Control[i]: val.update()
        for val in self.Center: val.update()
        self.mainAxis.update()
        self.Curve.update(Obj=self)

    def draw(self, screen, hide): # 오브젝트 그리기
        cx, cy = (self.Center[0].getPos(), self.Center[1].getPos())
        pygame.draw.circle(screen, BLACK, (cx,cy), r)
        if not hide: 
            for i in range(self.N):
                x, y = self.getControl(i)
                pygame.draw.circle(screen, BLUE, (x,y), r)
                pygame.draw.line(screen, GREY, (cx,cy), (x,y), LineWidth)
        self.Curve.draw()

    def getControl(self, i): # 제어점 위치 받기
        return self.Control[i][0].getPos(), self.Control[i][1].getPos()

    def getVelocity(self, i): # 제어점 속도 받기
        return self.Control[i][0].getSpeed(), self.Control[i][1].getSpeed()

    def getCenter(self): # 중심점 받기
        return self.Center[0].getPos(), self.Center[1].getPos()

    

class Simulation: # 오브젝트가 최종적으로 구현되는 Playground
    def __init__(self,  N, Mass, Length, k, aK, Damp, aDamp ,bonus, mvcN, mvcR):
        self.object = Object(N, Mass, Length, k, aK, Damp, aDamp, mvcN, mvcR)
        self.bonus = bonus

    def reset(self): 
        self.object.reset()

    def reward(self): # 강화학습을 위한 reward 계산
        Reward = 0
        for i in range(int(WIDTH / INTERVAL)):
            for j in range(int(HEIGHT / INTERVAL)):
                x, y = i * INTERVAL, j * INTERVAL
                if self.object.Curve.isInside(x, y): Reward += self.bonus[i][j]
        return Reward

    def render(self, screen, hide): # 화면에 표시
        screen.fill(WHITE)
        for i in range(int(WIDTH / INTERVAL)):
            for j in range(int(HEIGHT / INTERVAL)):
                x, y = i * INTERVAL, j * INTERVAL
                if self.bonus[i][j] == 0:
                    col = GREEN if self.object.Curve.isInside(x, y) else BLACK
                else:
                    col = BLUE if self.bonus[i][j] < 0 else RED
                pygame.draw.circle(screen, col, (x, y), 3)
        self.object.draw(screen, hide)

    def step(self, F): # 갱신
        self.object.extForce(F)
        self.object.calculate()
        self.object.update()
        return self.reward()


pygame.init() 
N, M, L, K, aK = 16, 1/10, 100, 7, 90000
DAMP, aDAMP = 0.2, 0.3
mvcN, mvcR = 50, 0.7*L
HIDEEXTRA = True # 여기에서 제어점 표시 여부 결정
bonus = [[0 for j in range(int(HEIGHT / INTERVAL))] for i in range(int(WIDTH / INTERVAL))]
simulation = Simulation(N, M, L, K, aK, DAMP, aDAMP, bonus, mvcN, mvcR)
screen = pygame.display.set_mode(Size)
pygame.display.set_caption("Simulation")
done = False
clock = pygame.time.Clock()
clock.tick(FPS)
simulation.reset()

time=0
while not done:
    time += 1
    F = [(0,  0) for i in range(N)]
    for i in range(int(N/2)):
        if time<150: continue
        F[i] = [-60,60] if time<350 else [0,0]
        F[(int(N/2)+i)%N] = [0,0] if time<400 else[60,-60]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    simulation.step(F)
    simulation.render(screen, HIDEEXTRA)
    pygame.display.flip()
