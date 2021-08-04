import math

import pygame #파이 게임 모듈 임포트

pygame.init() #파이 게임 초기화
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) #화면 크기 설정
clock = pygame.time.Clock()

#변수

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
r = 14


class Point:
    x = 0
    y = 0
    num = 0
    selected = 0

    def __init__(self, x, y, num=0):
        self.num = num
        self.x = x
        self.y = y

    def draw(self):
        if self.selected:
            pygame.draw.ellipse(screen, GREEN, [self.x-r/2, self.y-r/2, r, r], r)
        else:
            pygame.draw.ellipse(screen, BLACK, [self.x-r/2, self.y-r/2, r, r], r)

    def distance(self, p):
        return abs(math.sqrt((self.x-p.x)**2+(self.y-p.y)**2))


class Edge:
    p1 = Point(0, 0, 0)
    p2 = Point(0, 0, 0)

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self):
        pygame.draw.line(screen, GRAY, [self.p1.x, self.p1.y], [self.p2.x, self.p2.y], math.floor(r/2))


class MapHandler:
    preX = 0
    preY = 0
    preNum = -1
    x = 0
    y = 0
    pointList = []
    edgeList = []
    mouseClicked = False

    def export(self, file):
        arr = [[0]*len(self.pointList) for i in range(len(self.pointList))]

        file.write(repr(len(self.pointList)) + "\n")
        for point in self.pointList:
            file.write("[" + repr(point.x) + "," + repr(point.y) + "]")
            if self.pointList[-1] == point:
                file.write("\n")
            else:
                file.write(",\n")

        for edge in self.edgeList:
            arr[edge.p1.num][edge.p2.num] = edge.p1.distance(edge.p2)
            arr[edge.p2.num][edge.p1.num] = edge.p1.distance(edge.p2)

        file.write("\n")

        for i in range(0, len(self.pointList)):
            file.write("[")
            for j in range(0, len(self.pointList)):
                file.write(repr(arr[i][j]))
                if j != len(self.pointList) -1:
                    file.write(", ")

            file.write("]")
            if i == len(self.pointList) -1:
                file.write("\n")
            else:
                file.write(",\n")

        file.write("\n[")
        for point in self.pointList:
            file.write(repr(point.selected))
            if self.pointList[-1] != point:
                file.write(", ")
        file.write("]\n")

    def findpoint(self, x, y):
        i = 0
        for point in self.pointList:
            if point.distance(Point(x, y, 0)) < r:
                break
            i += 1
        return i

    def click(self, x, y, b):
        self.preX = x
        self.preY = y

        i = self.findpoint(x, y)

        if b == 3:
            if i == len(self.pointList):
                return
            elif self.pointList[i].selected == 0:
                self.pointList[i].selected = 1
            else:
                self.pointList[i].selected = 0
            return

        if i == len(self.pointList):
            p = Point(x, y, len(self.pointList))
            self.pointList.append(p)
            self.preNum = -1
            self.preX = -1
            self.preY = -1
            self.mouseClicked = False

        elif self.mouseClicked:
            if self.preNum != -1 and i < len(self.pointList):
                edge = Edge(self.pointList[self.preNum], self.pointList[i])
                self.edgeList.append(edge)
            self.mouseClicked = False
            self.preNum = -1
            self.preX = -1
            self.preY = -1

        else:
            self.preX = self.pointList[i].x
            self.preY = self.pointList[i].y
            self.preNum = self.pointList[i].num
            self.mouseClicked = True

    def clicking(self, x, y):
        pygame.draw.line(screen, GRAY, [self.preX, self.preY], [x, y], math.floor(r/2))

    def draw(self):
        for edge in self.edgeList:
            edge.draw()

        for point in self.pointList:
            point.draw()

        if self.mouseClicked:
            (x, y) = pygame.mouse.get_pos()
            self.clicking(x, y)

    def mouseclick(self, mouseevent):
        if mouseevent.type == pygame.MOUSEBUTTONUP:
            print("마우스 클릭")
            self.click(mouseevent.pos[0], mouseevent.pos[1], mouseevent.button)


mh = MapHandler()

while True:
    screen.fill(WHITE)

    event = pygame.event.poll() #이벤트 처리
    if event.type == pygame.QUIT:
        break
    mh.mouseclick(event)

    mh.draw()
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
txtFile = open("C:/Users/main-pc/Desktop", "w")
mh.export(txtFile)
