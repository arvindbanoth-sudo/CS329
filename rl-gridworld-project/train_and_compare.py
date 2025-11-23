# train_and_compare.py
import numpy as np
import matplotlib.pyplot as plt

from gridworld_env import GridWorld
from agents import QLearningAgent, SarsaAgent



def train_q(env, episodes = 500,
            epsilon_decay = 0.98):

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay
    )

    rewards = []
    lengths = []

    for ep in range(episodes):

        s = env.reset()
        total = 0
        steps = 0
        done = False

        while not done:
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)

            agent.update(s, a, r, s2, done)

            s = s2
            total += r
            steps += 1

        agent.decay_epsilon()
        rewards.append(total)
        lengths.append(steps)

    return agent, np.array(rewards), np.array(lengths)



def train_sarsa(env, episodes = 500,
                epsilon_decay = 0.98):

    agent = SarsaAgent(
        n_states = env.n_states,
        n_actions = env.n_actions,
        alpha=0.1, gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay
    )

    rewards = []
    lengths = []

    for ep in range(episodes):

        s = env.reset()
        a = agent.select_action(s)

        total=0
        steps=0
        done=False

        while not done:

            s2, r, done, _ = env.step(a)
            a2 = agent.select_action(s2) if not done else 0

            agent.update(s,a,r,s2,a2,done)

            s,a = s2, a2

            total += r
            steps += 1

        agent.decay_epsilon()
        rewards.append(total)
        lengths.append(steps)

    return agent, np.array(rewards), np.array(lengths)



def run_multiple(env, algo="q",
                 runs=5, episodes=500):

    all_rewards = []
    all_lengths = []
    last_agent  = None

    for _ in range(runs):

        if algo == "q":
            agent, r, l = train_q(env, episodes=episodes,
                                  epsilon_decay=0.98)
        else:
            agent, r, l = train_sarsa(env, episodes=episodes,
                                      epsilon_decay=0.98)

        last_agent = agent
        all_rewards.append(r)
        all_lengths.append(l)

    mean_rewards = np.mean(all_rewards, axis=0)
    mean_lengths = np.mean(all_lengths, axis=0)

    return last_agent, mean_rewards, mean_lengths



def moving_avg(x, w=50):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")



def print_policy(env, Q):
    arrows = {0:"↑",1:"↓",2:"←",3:"→"}

    grid = [["" for _ in range(env.width)] for _ in range(env.height)]

    for r in range(env.height):
        for c in range(env.width):

            pos = (r,c)
            if pos in env.obstacles:
                grid[r][c]="#"; continue
            if pos == env.goal:
                grid[r][c]="G"; continue

            s = env._state_to_idx(pos)
            a = np.argmax(Q[s])
            grid[r][c] = arrows[a]

    for row in grid: print(" ".join(row))
    print()



def main():

    env = GridWorld()

    print("Running multiple runs for Q-learning (for smoother curves)...")
    q_agent, q_r_mean, q_l_mean = run_multiple(env, "q",
                                               runs=5, episodes=500)

    print("Running multiple runs for SARSA...")
    s_agent, s_r_mean, s_l_mean = run_multiple(env, "s",
                                               runs=5, episodes=500)


    # ---- plots ----
    plt.figure()
    plt.title("Episode reward (mean over 5 runs)")
    plt.plot(q_r_mean, label="q-learning mean")
    plt.plot(s_r_mean, label="sarsa mean")
    plt.legend(); plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.savefig("rewards_raw.png")


    plt.figure()
    plt.title("Episode reward (moving avg, window=50)")
    plt.plot(moving_avg(q_r_mean), label="q-moving")
    plt.plot(moving_avg(s_r_mean), label="s-moving")
    plt.legend(); plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total reward (smoothed)")
    plt.savefig("rewards_moving_avg.png")


    plt.figure()
    plt.title("Episode lengths (mean over 5 runs)")
    plt.plot(q_l_mean, label="q-length")
    plt.plot(s_l_mean, label="s-length")
    plt.legend(); plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    plt.savefig("lengths.png")


    print("\nQ-Learning Policy (from last run):")
    print_policy(env, q_agent.Q)

    print("SARSA Policy (from last run):")
    print_policy(env, s_agent.Q)



if __name__ == "__main__":
    main()
