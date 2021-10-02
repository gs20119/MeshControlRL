import math
import pygame
import func as f
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH, HEIGHT = 800, 600
FPS = 120
ALPHA = 1 / FPS
r = 5
LineWidth = 5
INTERVAL = 25
Size = [WIDTH, HEIGHT]
stretchRange = 150
rotateRange = 20

# 이용할 색깔
BLACK = 0, 0, 0
WHITE = 255, 255, 255
BLUE = 0, 0, 255
GREEN = 0, 255, 0
RED = 255, 0, 0
GREY = 75, 75, 75


class Physics:  # 가속도, 각가속도가 여기에서 처리됨
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


class MVCCurve:  # Mean Value Coordinate 곡선을 처리
    def __init__(self, Obj, N, Radius):
        self.R0, self.N = Radius, N
        self.Weight = [[self.setWeight(Obj, i, j) for j in range(Obj.N)] for i in range(self.N)]
        self.Curve = [(0, 0) for i in range(self.N)]
        self.update(Obj)

    def setWeight(self, Obj, i, j):  # 초기 가중치 설정
        point = f.add(Obj.getCenter(), f.mul(self.R0, f.unitCircle(2 * math.pi * i / self.N)))
        prev = j - 1 if j > 0 else Obj.N - 1
        next = (j + 1) % Obj.N
        ret = math.tan(abs(f.angle(Obj.getControl(prev), Obj.getControl(j), point)) / 2)
        ret += math.tan(abs(f.angle(Obj.getControl(next), Obj.getControl(j), point)) / 2)
        ret /= f.dist(Obj.getControl(j), point)
        return ret

    def update(self, Obj):  # Obj를 따라 Curve 업데이트하기
        for i in range(self.N):
            fx, fy, totalW = 0, 0, 0
            for j in range(Obj.N):
                totalW += self.Weight[i][j]
                fx += Obj.getControl(j)[0] * self.Weight[i][j]
                fy += Obj.getControl(j)[1] * self.Weight[i][j]
            self.Curve[i] = fx / totalW, fy / totalW

    def draw(self, screen):  # Curve 그리기
        for i in range(self.N):
            x, y = self.Curve[i]
            pygame.draw.circle(screen, BLACK, (x, y), 2 * r / 3)

    def isInside(self, x, y):  # 어떤 점이 Curve 내부에 있는지 판정
        cross = 0
        for i in range(self.N):
            Pos1, Pos2 = self.Curve[i], self.Curve[(i + 1) % self.N]
            if (Pos1[1] > y and Pos2[1] <= y) or (Pos1[1] <= y and Pos2[1] > y):
                if Pos1[1] == Pos2[1]:
                    if x <= max(Pos1[0], Pos2[0]): cross += 1
                else:
                    atX = (y - Pos2[1]) * (Pos1[0] - Pos2[0]) / (Pos1[1] - Pos2[1]) + Pos2[0]
                    if x <= atX: cross += 1
        return cross % 2 > 0

    def outSide(self):
        for i in range(self.N):
            pos = self.Curve[i]
            if pos[0]<0 or pos[0]>WIDTH or pos[1]<0 or pos[1]>HEIGHT:
                return True
        return False


class Object:
    def __init__(self, N, Mass, Radius, k, aK, sK, damp, aDamp, mvcN, mvcR):
        self.N, self.mass, self.k, self.aK, self.sK, self.R0, self.damp, self.aDamp = (
            N, Mass, k, aK, sK, Radius, damp, aDamp)
        self.Control = [(Physics(self.R0 * math.cos(2 * math.pi * i / self.N) + WIDTH / 2),  # 제어점 (물리현상 근사)
                         Physics(self.R0 * math.sin(2 * math.pi * i / self.N) + HEIGHT / 2)) for i in range(self.N)]
        self.Center = [Physics(WIDTH / 2), Physics(HEIGHT / 2)]  # 오브젝트의 중심점
        self.mainAxis = Physics(0)  # 오브젝트의 주축
        self.totalI = (self.N * (self.R0 ** 2) * self.mass)  # 관성모멘트
        self.Tqnet = 0  # 알짜 토크
        self.Fnet = list([(0, 0) for i in range(self.N + 1)])  # 알짜힘
        for i in range(self.N + 1): self.Fnet[i] = list(self.Fnet[i])
        self.Curve = MVCCurve(self, mvcN, mvcR)  # MVC 곡선 (관찰의 대상)
        self.V0 = self.N * self.R0 * self.R0 * math.sin(2*math.pi / self.N)/2

    def reset(self):  # 오브젝트 초기화
        self.Control = [(Physics(self.R0 * math.cos(2 * math.pi * i / self.N) + WIDTH / 2),
                         Physics(self.R0 * math.sin(2 * math.pi * i / self.N) + HEIGHT / 2)) for i in
                        range(self.N)]
        self.Center = [Physics(WIDTH / 2), Physics(HEIGHT / 2)]
        self.mainAxis = Physics(0)
        self.V0 = self.N * self.R0 * self.R0 * math.sin(2 * math.pi / self.N)/2
        return self.getState()

    def getState(self):
        ret = list([0 for i in range(self.state_size())])
        for i in range(self.N):
            ret[2*i], ret[2*i+1] = self.getControl(i)
            ret[2*self.N + 2*i], ret[2*self.N + 2*i+1] = self.getVelocity(i)
        ret[4*self.N], ret[4*self.N+1] = self.getCenter()
        ret[4*self.N+2], ret[4*self.N+3] = self.Center[0].getSpeed(), self.Center[1].getSpeed()
        ret[4*self.N+4], ret[4*self.N+5] = self.mainAxis.getPos(), self.mainAxis.getSpeed()
        return ret

    def action_size(self):
        return 2*self.N

    def state_size(self):
        return 4 * self.N + 6

    def force(self, action):
        action = torch.squeeze(torch.FloatTensor(action).to(device))
        ret = [[0,0] for i in range(self.N)]
        for i in range(self.N):
            vec = f.dif(self.getControl(i), self.getCenter())
            theta = f.normal(vec) # action에서 stretch = 중심방향 힘, rotate = 접선방향 힘
            ret[i][0] = action[2*i].item()*theta[0]*stretchRange - action[2*i+1].item()*theta[1]*rotateRange
            ret[i][1] = action[2*i].item()*theta[1]*stretchRange + action[2*i+1].item()*theta[0]*rotateRange
        return ret

    def extForce(self, action):  # 외력 추가하기 (1)
        F = self.force(action)
        for i in range(self.N):
            self.Fnet[i][0] += F[i][0]
            self.Fnet[i][1] += F[i][1]

    def calculate(self):  # 탄성에 의한 힘과 토크 계산하기 (2)
        Center = self.getCenter()
        Area = list([0 for i in range(self.N)])
        sumArea = 0

        for i in range(self.N):
            pos = self.getControl(i)
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(self.k * (self.R0 - f.dist(pos, Center)), f.dir(Center, pos)))
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(-self.damp, self.getVelocity(i)))
            self.Fnet[self.N] = f.add(self.Fnet[self.N],
                                      f.mul(-self.k * (self.R0 - f.dist(pos, Center)), f.dir(Center, pos)))
            theta = math.atan2(pos[1] - self.getCenter()[1], pos[0] - self.getCenter()[0])
            Tq = self.aK * f.setArrange(theta - (2 * math.pi * i / self.N) - self.mainAxis.getPos(), -math.pi, math.pi,
                                        2 * math.pi)
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(Tq / (f.dist(pos, Center)),
                                                     f.unitCircle(theta - 1 * math.pi / 2)))#F=ma=m * alpha * r= m * T / (m*r^2) *r = T/r
            self.totalI += self.mass * (f.dist(pos, Center) ** 2)
            self.Tqnet += Tq
            Area[i] = 0.5 * f.dist(Center, self.getControl(i)) * f.dist(Center, self.getControl((i+1)%self.N)) * math.sin(f.angle(self.getControl(i), self.getControl((i+1)%self.N), Center))
            sumArea += Area[i]

        for i in range(self.N):
            self.Fnet[i] = f.add(self.Fnet[i], f.mul(self.sK*(self.V0-sumArea)*(Area[i]+Area[i-1]) /f.dist(Center, self.getControl(i)) , f.dir(Center, self.getControl(i))))

    def update(self):  # 상태 및 결과 업데이트하기 (3)
        for i in range(self.N):
            self.Control[i][0].accel(self.Fnet[i][0] / self.mass)
            self.Control[i][1].accel(self.Fnet[i][1] / self.mass)
        self.Center[0].accel(self.Fnet[self.N][0] / (self.N * self.mass))
        self.Center[1].accel(self.Fnet[self.N][1] / (self.N * self.mass))
        self.mainAxis.accel(self.Tqnet / self.totalI)
        for i in range(self.N + 1): self.Fnet[i][0] = self.Fnet[i][1] = 0
        self.totalI = self.Tqnet = 0

        for i in range(self.N):
            for val in self.Control[i]: val.update()
        for val in self.Center: val.update()
        self.mainAxis.update()
        self.Curve.update(self)

    def draw(self, screen, hide):  # 오브젝트 그리기
        cx, cy = (self.Center[0].getPos(), self.Center[1].getPos())
        pygame.draw.circle(screen, BLACK, (cx, cy), r)
        if not hide:
            for i in range(self.N):
                x, y = self.getControl(i)
                pygame.draw.circle(screen, BLUE, (x, y), r)
                pygame.draw.line(screen, GREY, (cx, cy), (x, y), LineWidth)
        self.Curve.draw(screen)

    def getControl(self, i):  # 제어점 위치 받기
        if i<0 :
            i += self.N
        elif i>=self.N :
            i -= self.N
        return self.Control[i][0].getPos(), self.Control[i][1].getPos()

    def getVelocity(self, i):  # 제어점 속도 받기
        return self.Control[i][0].getSpeed(), self.Control[i][1].getSpeed()

    def getCenter(self):  # 중심점 받기
        return self.Center[0].getPos(), self.Center[1].getPos()

class Simulation:  # 오브젝트가 최종적으로 구현되는 Playground
    def __init__(self, N, Mass, Length, k, aK, sK, Damp, aDamp, bonus, mvcN, mvcR, screen):
        self.object = Object(N, Mass, Length, k, aK, sK, Damp, aDamp, mvcN, mvcR)

        self.bonus = bonus
        self.screen = screen
        self.pre = self.reward()

    def reset(self):
        ret = self.object.reset()
        self.pre = self.reward()
        return ret

    def reward(self):  # 강화학습을 위한 reward 계산
        Reward = 0
        for i in range(int(HEIGHT / INTERVAL)):
            for j in range(int( WIDTH/ INTERVAL)):
                y, x = i * INTERVAL, j * INTERVAL
                if self.object.Curve.isInside(x, y):
                    Reward += self.bonus[i][j]
        return Reward/10

    def render(self,  hide):  # 화면에 표시
        self.screen.fill(WHITE)
        for i in range(int(HEIGHT / INTERVAL)):
            for j in range(int(WIDTH / INTERVAL)):
                x, y = j * INTERVAL, i * INTERVAL
                if self.bonus[i][j] == 0:
                    col = GREEN if self.object.Curve.isInside(x, y) else BLACK
                else:
                    col = BLUE if self.bonus[i][j] < 0 else RED
                pygame.draw.circle(self.screen, col, (x, y), 3)
        self.object.draw(self.screen, hide)

    def step(self, action):  # 갱신
        self.object.extForce(action)
        self.object.calculate()
        self.object.update()
        reward = self.reward()
        ret = reward - self.pre
        self.pre = reward
        done = self.object.Curve.outSide()
        if not done:
            return self.getState(), ret, False
        return self.getState(), -10, True

    def action_size(self):
        return self.object.action_size()

    def state_size(self):
        return self.object.state_size()

    def getState(self):
        return self.object.getState()
