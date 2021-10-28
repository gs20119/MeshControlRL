import math

def add(a, b):
    c = [a[i]+b[i] for i in range(len(a))]
    return c

def mul(c, a):
    r = [c * a[i] for i in range(len(a))]
    return r

def dotProduct(v1, v2):
    Res = 0
    for a, b in zip(v1, v2): Res += a*b
    return Res

def length(v):
  return math.sqrt(dotProduct(v, v))

def normal(v):
    return mul(1/length(v),v)

def angleOfVector(v1, v2):
    temp = dotProduct(v1, v2) / (length(v1) * length(v2))
    if temp >1 :
        temp =1
    elif temp <-1:
        temp = -1
    return math.acos(temp)


def angle(p1, p2, o):
    p1 = list(p1)
    p2 = list(p2)
    for i in range(len(o)):
        p1[i] -= o[i]
        p2[i] -= o[i]

    return angleOfVector(p1, p2)

def unitCircle(theta):
    return (math.cos(theta), math.sin(theta))

def vectorProduct(v1, v2):
    return v1[0]*v2[1]-v2[0]*v1[1]

def changeDegree(v, theta):
    return (math.cos(theta)*v[0] -math.sin(theta)*v[1], math.sin(theta)*v[0]+math.cos(theta)*v[1])

def setArrange(value, start, end, gap):
    while value < start:
        value += gap
    while value > end:
        value -= gap
    return value

def dif(v1, v2):
    return add(v2, mul(-1, v1))

def dist(v1, v2) :
    return length(dif(v1, v2))

def dir(v1, v2):
    return mul(1/dist(v1, v2), dif(v1, v2))