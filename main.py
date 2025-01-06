import math
import random

import cv2
import numpy as np
import random as rnd
from numba.typed import List
import numba
from time import perf_counter


SCREEN = (680, 1200)
CENTER = (SCREEN[0]/2, SCREEN[1]/2)
MARGIN = 10
REFRESH_RATE = 25

PI = math.pi

AGENTS = 100
SPAWN_AREA = 100
EVAPORATION_CONSTANT = 0.0001
EVAPORATION_COEF = 1.025

PROB_TABLE_SIZE = List([75, 1])

@numba.jit()
def construct(n, prob):
    stack = List()
    probabilities = List()
    for agent in range(n):
        pos = List([rnd.randint(-SPAWN_AREA,SPAWN_AREA), rnd.randint(-SPAWN_AREA,SPAWN_AREA)])
        dir = List([rnd.randint(-1,1), rnd.randint(-1,1)])
        stack.append(List([pos, dir]))

    for x in range(prob[0]):
        if x > prob[1] - 1:
            probabilities.append(False)
        else:
            probabilities.append(True)

    return stack[2:], probabilities


@numba.jit()
def drawAgents(img, stack):
    for agent in stack:
        img[int(agent[0][1]+CENTER[0]), int(agent[0][0]+CENTER[1])] = 255

    return img

@numba.jit()
def move(stack, prob):

    for i, agent in enumerate(stack):
        x = random.randint(0, len(prob)-1)
        chance = prob[x]

        if chance:
            agent[1][1] = rnd.randint(-1, 1)
            agent[1][0] = rnd.randint(-1, 1)

        x = agent[0][0]+agent[1][0]
        y = agent[0][1]+agent[1][1]

        if x >= CENTER[1] - MARGIN:
            agent[1][0] = -1
            agent[1][1] = rnd.randint(-1, 1)
        elif x <= -CENTER[1] + MARGIN:
            agent[1][0] = 1
            agent[1][1] = rnd.randint(-1, 1)

        if y >= CENTER[0] - MARGIN:
            agent[1][1] = -1
            agent[1][0] = rnd.randint(-1, 1)
        elif y <= -CENTER[0] + MARGIN:
            agent[1][1] = 1
            agent[1][0] = rnd.randint(-1, 1)

        x = agent[0][0] + agent[1][0]
        y = agent[0][1] + agent[1][1]

        stack[i][0] = List([x, y])

    return stack


@numba.jit()
def blur(img):
    height = len(img)
    width = len(img[0])


    for x in range(height):
        for y in range(width):
            if x < 1 or y < 1 or (x + 1) == height or (y + 1) == width:
                continue

            sum = img[x - 1, y + 1] + \
                img[x + 0, y + 1] + \
                img[x + 1, y + 1] + \
                img[x - 1, y + 0] + \
                img[x + 0, y + 0] + \
                img[x + 1, y + 0] + \
                img[x - 1, y - 1] + \
                img[x + 0, y - 1] + \
                img[x + 1, y - 1]

            img[x, y] = sum/(9*EVAPORATION_COEF)
    return img


@numba.jit()
def evaporate(img):
    height = len(img)
    width = len(img[0])

    for x in range(height):
        for y in range(width):
            if img[x,y] != 0:
                img[x,y] -= EVAPORATION_CONSTANT
    return img


canvas = np.zeros((SCREEN[0], SCREEN[1]))
cv2.startWindowThread()
agentStack, prob = construct(AGENTS, PROB_TABLE_SIZE)


while(True):
    cv2.imshow('',canvas)
    cv2.waitKey(REFRESH_RATE) # here you specify refresh rate in ms
    timeit = perf_counter()
    agentStack = move(agentStack, prob)
    canvas = drawAgents(canvas, agentStack)
    canvas = evaporate(canvas)
    canvas = blur(canvas)
    print(perf_counter() - timeit)


