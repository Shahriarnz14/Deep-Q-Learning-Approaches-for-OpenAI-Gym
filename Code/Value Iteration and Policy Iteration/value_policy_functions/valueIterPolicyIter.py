# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import deeprl_hw2q2.lake_envs as lake_env
import gym
import matplotlib.pyplot as plt
import seaborn as sns

def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """

    policy = np.zeros(env.nS, dtype='int')

    for state in range(env.nS):

        argmax_action = []
        max_action_value = -np.inf

        for action in range(env.nA):
            [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
            max_action_value_tmp = reward + gamma * value_function[nextstate]
            if max_action_value_tmp > max_action_value:
                max_action_value = max_action_value_tmp
                argmax_action = action

        policy[state] = argmax_action

    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)
        for state in range(env.nS):
            v = value_func[state]
            [(prob, nextstate, reward, is_terminal)] = env.P[state][policy[state]]
            value_func_tmp[state] = reward + gamma * value_func[nextstate]
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)
        for state in range(env.nS):
            v = value_func_tmp[state]
            [(prob, nextstate, reward, is_terminal)] = env.P[state][policy[state]]
            value_func_tmp[state] = reward + gamma * value_func_tmp[nextstate]
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)

        state_rand_perm = np.random.permutation(env.nS)
        # print(state_rand_perm)
        for state in state_rand_perm:
            v = value_func_tmp[state]
            [(prob, nextstate, reward, is_terminal)] = env.P[state][policy[state]]
            value_func_tmp[state] = reward + gamma * value_func_tmp[nextstate]
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    policy_stable = True
    for state in range(env.nS):
        old_action = policy[state]
        argmax_action = []
        max_action = -np.inf
        for action in range(env.nA):
            [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
            max_action_tmp = prob * (reward + gamma * value_func[nextstate])
            if max_action_tmp > max_action:
                max_action = max_action_tmp
                argmax_action = action
        policy[state] = argmax_action
        if old_action != argmax_action:
            policy_stable = False

    return (not policy_stable), policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.
	
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    num_policy_iter = 0
    num_policy_imp = 0
    is_stable = True
    while is_stable:
        value_func, num_value_iter = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)
        is_stable, policy = improve_policy(env, gamma, value_func, policy)

        num_policy_iter += num_value_iter
        num_policy_imp += 1

    # for state in range(env.nS):
    #     for action in range(env.nA):
    #         [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
    #         print('Prob:', prob, ' - State:', state, ' - Action', action)

    return policy, value_func, num_policy_imp, num_policy_iter


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.
	

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    num_policy_iter = 0
    num_policy_imp = 0
    is_stable = True
    while is_stable:
        value_func, num_value_iter = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)
        is_stable, policy = improve_policy(env, gamma, value_func, policy)

        num_policy_iter += num_value_iter
        num_policy_imp += 1

    return policy, value_func, num_policy_imp, num_policy_iter


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

	
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    num_policy_iter = 0
    num_policy_imp = 0
    is_stable = True
    while is_stable:
        value_func, num_value_iter = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)
        is_stable, policy = improve_policy(env, gamma, value_func, policy)

        num_policy_iter += num_value_iter
        num_policy_imp += 1

    return policy, value_func, num_policy_imp, num_policy_iter


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)
        for state in range(env.nS):
            v = value_func[state]
            max_val = -np.inf
            for action in range(env.nA):
                [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
                max_val_tmp = reward + gamma * value_func[nextstate]
                if max_val_tmp > max_val:
                    max_val = max_val_tmp
            value_func_tmp[state] = max_val
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)
        for state in range(env.nS):
            v = value_func_tmp[state]
            max_val = -np.inf
            for action in range(env.nA):
                [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
                max_val_tmp = reward + gamma * value_func_tmp[nextstate] #should it be value_func_tmp or value_func
                if max_val_tmp > max_val:
                    max_val = max_val_tmp
            value_func_tmp[state] = max_val
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)

        state_rand_perm = np.random.permutation(env.nS)
        # print(state_rand_perm)
        for state in state_rand_perm:
            v = value_func_tmp[state]
            max_val = -np.inf
            for action in range(env.nA):
                [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
                max_val_tmp = reward + gamma * value_func_tmp[nextstate] #should it be value_func_tmp or value_func
                if max_val_tmp > max_val:
                    max_val = max_val_tmp
            value_func_tmp[state] = max_val
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    # Assume Square-Grid Environment

    # Find the Goal state
    goal_state = -1
    for state in range(env.nS):
        for action in range(env.nA):
            [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
            if reward == 1:
                goal_state = nextstate
    # print(goal_state)

    # Find distance of each state to the Goal state
    states_distance_to_goal = np.zeros((env.nS,))
    for state in range(env.nS):
        states_distance_to_goal[state] = np.sum(np.abs(np.array(divmod(goal_state, env.ncol)) -
                                                       np.array(divmod(state, env.ncol))))



    manhattan_distance_ordered_states = np.argsort(states_distance_to_goal)
    # print(manhattan_distance_ordered_states)

    num_iteration = 0
    while True:
        delta = 0
        value_func_tmp = np.copy(value_func)
        for state in manhattan_distance_ordered_states:
            v = value_func_tmp[state]
            max_val = -np.inf
            for action in range(env.nA):
                [(prob, nextstate, reward, is_terminal)] = env.P[state][action]
                max_val_tmp = reward + gamma * value_func_tmp[nextstate] 
                if max_val_tmp > max_val:
                    max_val = max_val_tmp
            value_func_tmp[state] = max_val
            delta = np.max([delta, np.abs(v - value_func_tmp[state])])
        num_iteration += 1
        if delta < tol or num_iteration > max_iterations:
            break
        value_func = np.copy(value_func_tmp)

    value_func = np.copy(value_func_tmp)

    return value_func, num_iteration

    return value_func, 0


######################
#  Optional Helpers  #
######################

def display_policy_letters(env, policy):
    """Displays a policy as letters

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
        for action in range(env.nA):
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                env.T[state, action, nextstate] = prob
                env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap

    Note that you need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels=np.arange(1, env.nrow + 1)[::-1],
                xticklabels=np.arange(1, env.nrow + 1))
    plt.show()
    return None
