
import sys
import math
from simulation import *
from td3 import *
from collections import deque
import matplotlib.pyplot as plt

o, X = 1, -1
bonus = [
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, o, o, o, o, o, o, o, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
[X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
]

N, M, L, K, aK, sK = 16, 1 / 16, 100, 7, 90000, 0.0
DAMP, aDAMP = 0.3, 0.3
mvcN, mvcR = 50, 0.7 * L
HIDEEXTRA = False  # 여기에서 제어점 표시 여부 결정

pygame.init()
screen = pygame.display.set_mode([800, 600])
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()
clock.tick(FPS)
simulation = Simulation(N, M, L, K, aK, sK, DAMP, aDAMP, bonus, mvcN, mvcR, screen)
done = False

train_mode = True
load_model = False

num_epochs = 50000
max_step = 500
discount_factor = 0.99
step = 0
episode = 0
start_train_episode = 10
print_interval = 10
actor_loss = None
critic_loss = None

state_size = simulation.state_size()
action_size = simulation.action_size()

agent = TD3(state_size, action_size)

state = simulation.reset()
done = False
step = 0

for episode in range(num_epochs) :
    step =0
    done = False
    state = simulation.reset()

    while step<max_step and (done==False):
        step += 1
        action = agent.get_action([state])
        #print(action)
        next_state, reward, done = simulation.step(action)

        if train_mode:
            agent.append_sample(state, action, reward, next_state, done)

        state = next_state
        if train_mode and episode >= start_train_episode:
            actor_loss, critic_loss = agent.train()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        simulation.render(HIDEEXTRA)
        pygame.display.flip()

        if step % print_interval == 0:
            print("episode : {} / step: {} / actor_loss: {} / critic_loss: {} / value: {}".format(episode, step, actor_loss, critic_loss, reward))
            #print("main action : {}, {}".format(action[2*N],action[2*N+1]))
