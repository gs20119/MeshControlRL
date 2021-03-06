
import sys
import math
from simulation import *
from td3 import *
from collections import deque
import matplotlib.pyplot as plt

X, O = 1, -1
bonus = [
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, X, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O],
]

N, M, L, K, aK, sK = 32, 1/32, 100, 7.5, 90000, 0.0
DAMP, aDAMP = 0.2, 0.3
mvcN, mvcR = 50, 0.7*L
HIDEEXTRA = True  # ???????????? ????????? ?????? ?????? ??????
HIDEBOARD = False

pygame.init()
screen = pygame.display.set_mode([800, 800])
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()
clock.tick(FPS)
simulation = Simulation(N, M, L, K, aK, sK, DAMP, aDAMP, bonus, mvcN, mvcR, screen)
done = False

train_mode = True
load_model = False

num_epochs = 50000
max_step = 1000
discount_factor = 0.99
step = 0
episode = 0
start_train_episode = 2
print_interval = 10
actor_loss = None
critic_loss = None

state_size = simulation.state_size()
action_size = simulation.action_size()

agent = TD3(state_size, action_size)

state = simulation.reset()
done = False
ended = False
step = 0
critic_losses = deque(maxlen=100)
actor_losses = deque(maxlen=100)
critic_average = []
actor_average = []
sum_actor = 0
sum_critic = 0

for episode in range(num_epochs):
    step = 0
    done = False
    state = simulation.reset()
    while step<max_step and (done==False):
        step += 1
        action = agent.get_action([state])

        print(action[1:6])
        next_state, reward, done = simulation.step(action)

        if train_mode:
            agent.append_sample(state, action, reward, next_state, done)

        state = next_state
        if train_mode and episode >= start_train_episode:
            actor_loss, critic_loss = agent.train()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            sum_actor += actor_loss
            sum_critic += critic_loss
            if len(critic_losses) == 100:
                sum_actor -= actor_losses[0]
                sum_critic -= critic_losses[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ended=True

        simulation.render(HIDEEXTRA, HIDEBOARD)
        pygame.display.flip()

        if step % print_interval == 0:
            print("episode : {} / step: {} / actor_loss: {} / critic_loss: {} / value: {}".format(episode, step, actor_loss, critic_loss, reward))
            #print("main action : {}, {}".format(action[2*N],action[2*N+1]))   

    actor_average.append(sum_actor/100)
    critic_average.append(sum_critic/100) 

    if ended:
        break

plt.figure()
plt.plot(actor_average)
plt.figure()
plt.plot(critic_average)
plt.show()