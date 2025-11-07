"""Try gym dynamics with LQR using linearized A,B matrices"""
import numpy as np
import gymnasium as gym
from scipy.linalg import solve_continuous_are


# Set up cartpole env
env = gym.make("CartPole-v1", render_mode="human")

# Set up linearized dynamics for LQR

mp = 0.1 # kg
mc = 1.0 # kg
g = 9.81 # m/s^2
l = 0.5 # m

A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 3*mp*g/(4*(mp+mc)-3*mp), 0, 0], [0, g/(l*((4.0/3.0) - (mp/(mp+mc)))), 0, 0]])
B = np.array([[0], [0], [1/(mc+mp)], [(-1/(mc+mp)) * (1/(l*((4.0/3.0) - (mp/(mp+mc)))))]])

# state matrix
mk = mc
lp = l
mt = mc + mp
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mk)))
B = np.array([[0], [1/mt], [0], [b]])
Q = np.diag([1, 10, 1, 1])
R = np.eye(1)

P = solve_continuous_are(A, B, Q, R)
K = -np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
# K = np.dot(np.linalg.inv(R),
#            np.dot(B.T, P))
observation, info = env.reset()

episode_over = False
total_reward = 0

max_abs_force = 0

while not episode_over:
    x = observation
    u = K @ x

    if u < 0:
        action = 0
    else:
        action = 1

    abs_force = np.abs(float(np.clip(*u, -10, 10)))

    if abs_force > max_abs_force:
        max_abs_force = abs_force


    env.force_mag = 10

    

    observation, reward, terminated, truncated, info = env.step(1)

    total_reward += reward
    episode_over = terminated or truncated

print(observation)
print(f"Episodee finished! Total Reward: {total_reward}. Max Force: {max_abs_force} N")
env.close()