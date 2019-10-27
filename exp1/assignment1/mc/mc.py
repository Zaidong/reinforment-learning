import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

#create env
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def mc(env, num_episodes, discount_factor=1.0, epsilon=0.1, first_visit=1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

#############################################Implement your code###################################################################################################
        episode = []
        done = False
        state = env.reset()
        while not done:
            a_p = policy(state)
            temp = np.random.uniform(0, 1)
            # select an action based on the possibility array
            for i in range(env.action_space.n):
                temp -= a_p[i]
                if temp < 0:
                    action = i
                    break

            #print(action)
            state_, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = state_
        if first_visit:
            returns_count_one_episode = defaultdict(float)
            for i, exp in enumerate(episode):
                if returns_count_one_episode[exp[0]] == 0:
                    returns_count_one_episode[exp[0]] += 1
                    G = sum([e[2] * discount_factor ** (j - i) for j, e in enumerate(episode[i:])])
                    returns_sum[exp[0]][exp[1]] += G
                    returns_count[exp[0]][exp[1]] += 1
                    Q[exp[0]][exp[1]] = returns_sum[exp[0]][exp[1]] / returns_count[exp[0]][exp[1]]
        else:
            for i, exp in enumerate(episode):
                G = sum([e[2] * discount_factor ** (j - i) for j, e in enumerate(episode[i:])])
                returns_sum[exp[0]][exp[1]] += G
                returns_count[exp[0]][exp[1]] += 1
                Q[exp[0]][exp[1]] = returns_sum[exp[0]][exp[1]] / returns_count[exp[0]][exp[1]]
 #############################################Implement your code end###################################################################################################
    return Q, policy


Q_first, policy_first = mc(env, num_episodes=500000, epsilon=0.1)
Q_every, policy_every = mc(env, num_episodes=500000, epsilon=0.1, first_visit=0)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V_first = defaultdict(float)
V_every = defaultdict(float)
for state, actions in Q_first.items():
    action_value = np.max(actions)
    V_first[state] = action_value
for state, actions in Q_every.items():
    action_value = np.max(actions)
    V_every[state] = action_value
plotting.plot_value_function(V_first, title="(first_visit)Optimal Value Function")
plotting.plot_value_function(V_every, title="(every_visit)Optimal Value Function")
