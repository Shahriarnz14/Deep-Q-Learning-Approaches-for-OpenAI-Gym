#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import value_policy_functions.lake_envs as lake_env
import value_policy_functions.rl as rl
import time
import numpy as np


def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so we can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    # print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)

    env.render()

    random_permutation_trial = 10

    run_model = input('0: Random Policy, 1: Policy-Iter, 2: Value-Iter, \n 3: Policy-Iter-Order, 4: Policy-Iter-Rnd'
                      + '5: Value-Iter-Order, 6: Value-Iter-Rnd \n 7: Custom\n\n')

    run_model = int(run_model)

    if run_model == 0:
        [(p, ns, w, aa)] = env.P[4][0]
        input('Hit enter to run a random policy...')

        total_reward, num_steps = run_random_policy(env)
        print('Agent received total reward of: %f' % total_reward)
        print('Agent took %d steps' % num_steps)

    elif run_model == 1:
        print('Policy Iteration Sync...\n')
        policy, value_func, num_policy_imp, num_policy_iter = rl.policy_iteration_sync(env, gamma=0.9,
                                                                                       max_iterations=int(1e3),
                                                                                       tol=1e-3)
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)

        print('#Policy Improvements: ' + str(num_policy_imp))
        print('#Total Iterations: ' + str(num_policy_iter))
        # rl.print_policy(policy, lake_env.action_names)

    elif run_model == 2:
        print('Value Iteration Sync...\n')
        value_func, num_iteration = rl.value_iteration_sync(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        policy = rl.value_function_to_policy(env, gamma=0.9, value_function=value_func)
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print('Total Iterations: ' + str(num_iteration))

    elif run_model == 3:
        print('Policy Iteration Async - Ordered ...\n')
        policy, value_func, num_policy_imp, num_policy_iter = rl.policy_iteration_async_ordered(env, gamma=0.9,
                                                                                                max_iterations=int(1e3),
                                                                                                tol=1e-3)
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print('#Policy Improvements: ' + str(num_policy_imp))
        print('#Total Iterations: ' + str(num_policy_iter))
        # rl.print_policy(policy, lake_env.action_names)

    elif run_model == 4:
        print('Policy Iteration Async - Random Permutation ...\n')
        policy_improvements_count_trials = np.zeros((random_permutation_trial,))
        total_iterations_count_trials = np.zeros((random_permutation_trial,))
        for trial_rnd_perm_idx in range(random_permutation_trial):
            policy, value_func, num_policy_imp, num_policy_iter = rl.policy_iteration_async_randperm(env, gamma=0.9,
                                                                                                     max_iterations=int(
                                                                                                         1e3), tol=1e-3)
            policy_improvements_count_trials[trial_rnd_perm_idx] = num_policy_imp
            total_iterations_count_trials[trial_rnd_perm_idx] = num_policy_iter

        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print(total_iterations_count_trials)
        print('#Average Policy Improvements: ' + str(np.mean(policy_improvements_count_trials)))
        print('#Average Total Iterations: ' + str(np.mean(total_iterations_count_trials)))
        # rl.print_policy(policy, lake_env.action_names)

    elif run_model == 5:
        print('Value Iteration Async - Ordered ...\n')
        value_func, num_iteration = rl.value_iteration_async_ordered(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        policy = rl.value_function_to_policy(env, gamma=0.9, value_function=value_func)
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print('Total Iterations: ' + str(num_iteration))

    elif run_model == 6:
        print('Value Iteration Async - Random Permutation ...\n')
        total_iterations_count_trials = np.zeros((random_permutation_trial,))
        for trial_rnd_perm_idx in range(random_permutation_trial):
            value_func, num_iteration = rl.value_iteration_async_randperm(env, gamma=0.9, max_iterations=int(1e3),
                                                                          tol=1e-3)
            policy = rl.value_function_to_policy(env, gamma=0.9, value_function=value_func)
            total_iterations_count_trials[trial_rnd_perm_idx] = num_iteration
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print('Average Total Iterations: ' + str(np.mean(total_iterations_count_trials)))

    elif run_model == 7:
        print('Value Iteration Async - Manhattan Distance ...\n')
        value_func, num_iteration = rl.value_iteration_async_custom(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        policy = rl.value_function_to_policy(env, gamma=0.9, value_function=value_func)
        rl.display_policy_letters(env, policy)
        rl.value_func_heatmap(env, value_func)
        print('Total Iterations: ' + str(num_iteration))

    else:
        print('Nothing Performed!\n')

    # print_env_info(env)
    # total_reward, num_steps = run_random_policy(env)
    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
