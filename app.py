import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def main():
    env = gym.make("Acrobot-v1")

    n_actions = env.action_space.n
    observation_size = env.observation_space.shape[0]

    print("Observation size:", observation_size)
    print("Number of actions:", n_actions)

    # Acrobot has six continuous observation values:

    # 0: cos(theta1)
    # 1: sin(theta1)
    # 2: cos(theta2)
    # 3: sin(theta2)
    # 4: angular velocity of theta1
    # 5: angular velocity of theta2

    # Each continuous value is divided into discrete bins so that it can be used as an index in the Q-table
    state_bins = (
        8,   # cos(theta1)
        8,   # sin(theta1)
        8,   # cos(theta2)
        8,   # sin(theta2)
        10,  # angular velocity theta1
        10   # angular velocity theta2
    )

    state_low = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -4.0 * np.pi,
            -9.0 * np.pi
        ],
        dtype=np.float32
    )

    state_high = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            4.0 * np.pi,
            9.0 * np.pi
        ],
        dtype=np.float32
    )

    discretization_bins = create_discretization_bins(
        state_low=state_low,
        state_high=state_high,
        state_bins=state_bins
    )

    # One additional dimension is needed for the actions
    q_table_shape = state_bins + (n_actions,)

    # Small random initial values help prevent every action from initially having exactly the same Q-value
    rng = np.random.default_rng()

    q_table = rng.uniform(
        low=-0.01,
        high=0.01,
        size=q_table_shape
    ).astype(np.float32)

    print("Q-table shape:", q_table.shape)
    print("Number of Q-values:", q_table.size)

    # Q-learning hyperparameters
    learning_rate = 0.10
    gamma = 0.99

    initial_epsilon = 1.0
    minimum_epsilon = 0.05
    epsilon_decay = 0.995

    epsilon = initial_epsilon

    HORIZON = 500
    MAX_TRAJECTORIES = 1000

    scores = []
    rewards = []
    td_errors = []
    successes = []
    epsilon_history = []

    training_start_time = time.perf_counter()

    for trajectory in range(MAX_TRAJECTORIES):
        # Step 1: Initialize the starting state
        current_observation, info = env.reset()

        current_state = discretize_state(
            observation=current_observation,
            discretization_bins=discretization_bins
        )

        total_reward = 0.0
        total_absolute_td_error = 0.0
        step_count = 0

        terminated = False
        truncated = False

        for step in range(HORIZON):
            # Step 2: Select an action using epsilon-greedy
            action = select_action(
                q_table=q_table,
                state=current_state,
                epsilon=epsilon,
                n_actions=n_actions,
                rng=rng
            )

            # Step 3: Execute the action and observe:nreward, next state, and terminal status
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated

            next_state = discretize_state(
                observation=next_observation,
                discretization_bins=discretization_bins
            )

            current_q_value = q_table[current_state + (action,)]

            # Step 4: Calculate the Q-learning target
            # Terminal: target = reward
            # Non-terminal: target = reward + gamma * max Q(s', a')
            if terminated:
                target_q_value = reward
            else:
                best_next_q_value = np.max(q_table[next_state])

                target_q_value = (
                    reward
                    + gamma * best_next_q_value
                )

            # Step 5: Calculate the TD error.
            # delta = target - Q(s, a)
            td_error = target_q_value - current_q_value

            # Step 6: Update the Q-table.
            # Q(s, a) <- Q(s, a) + alpha * TD error
            q_table[current_state + (action,)] += (
                learning_rate * td_error
            )

            total_reward += reward
            total_absolute_td_error += abs(td_error)
            step_count += 1

            if done:
                break

            # Step 7: Move to the next state
            current_state = next_state

        # Reduce exploration after every trajectory
        epsilon = max(
            minimum_epsilon,
            epsilon * epsilon_decay
        )

        scores.append(step_count)
        rewards.append(total_reward)

        average_episode_td_error = (
            total_absolute_td_error / max(step_count, 1)
        )

        td_errors.append(average_episode_td_error)
        epsilon_history.append(epsilon)

        # Acrobot terminates successfully when the free end reaches the required height. A truncated trajectory reached the time limit and is not counted as success
        success = terminated and not truncated
        successes.append(success)

        completed_trajectories = trajectory + 1

        if completed_trajectories % 50 == 0:
            average_score = np.mean(scores[-50:])
            average_reward = np.mean(rewards[-50:])
            success_rate = np.mean(successes[-50:]) * 100
            average_td_error = np.mean(td_errors[-50:])

            elapsed_time = (
                time.perf_counter()
                - training_start_time
            )

            print(
                f"Trajectory {completed_trajectories}\t"
                f"Average Score: {average_score:.2f}\t"
                f"Average Reward: {average_reward:.2f}\t"
                f"Success Rate: {success_rate:.2f}%\t"
                f"Average TD Error: {average_td_error:.4f}\t"
                f"Epsilon: {epsilon:.4f}\t"
                f"Training Time: {elapsed_time:.2f}s"
            )

    training_time = (
        time.perf_counter()
        - training_start_time
    )

    env.close()

    print("\nTraining complete.")
    print(f"Total training time: {training_time:.2f} seconds")
    print(
        "Final 50-trajectory success rate: "
        f"{np.mean(successes[-50:]) * 100:.2f}%"
    )
    print(
        "Final 50-trajectory average score: "
        f"{np.mean(scores[-50:]):.2f}"
    )

    score_array = np.array(scores)

    # Tracking 3 performance-related metrics:
        # 1. Number of Steps per Trajectory
        # 2. Success Rate %
        # 3. Error Rate
    generate_scatter_plot(score_array)
    generate_success_rate_plot(np.array(successes))
    generate_td_error_plot(np.array(td_errors))


def create_discretization_bins(state_low, state_high, state_bins):
    # Creates the boundaries used to convert each continuous observation value into a discrete integer
    
    discretization_bins = []

    for dimension in range(len(state_bins)):
        boundaries = np.linspace(
            state_low[dimension],
            state_high[dimension],
            state_bins[dimension] + 1
        )[1:-1]

        discretization_bins.append(boundaries)

    return discretization_bins


def discretize_state(observation,discretization_bins):
    # Converts a continuous Acrobot observation into a tuple of integer Q-table indexes

    discrete_state = []

    for dimension, boundaries in enumerate(discretization_bins):
        discrete_value = np.digitize(
            observation[dimension],
            boundaries
        )

        discrete_state.append(discrete_value)

    return tuple(discrete_state)


def select_action(q_table, state, epsilon, n_actions, rng):
    # Selects an action using the epsilon-greedy strategy.

    # Exploration: Select a random action
    # Exploitation: Select an action with the highest Q-value
    
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    state_q_values = q_table[state]
    maximum_q_value = np.max(state_q_values)

    # Randomly choose between tied best actions rather than always selecting the first action returned by argmax
    best_actions = np.flatnonzero(
        np.isclose(
            state_q_values,
            maximum_q_value
        )
    )

    return int(rng.choice(best_actions))


def generate_scatter_plot(score):
    # Plots the number of steps taken during each trajectory
    # Lower trajectory duration indicates better performance for Acrobot as the goal is to reach the target height in as few steps as possible

    average_score = running_mean(score)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Trajectory Duration", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    plt.plot(
        np.arange(len(score)),
        score,
        color="gray",
        linewidth=1,
        label="Trajectory duration"
    )

    plt.scatter(
        np.arange(len(score)),
        score,
        color="green",
        linewidth=0.3,
        label="Individual trajectory"
    )

    if len(average_score) > 0:
        average_x = np.arange(
            49,
            49 + len(average_score)
        )

        plt.plot(
            average_x,
            average_score,
            color="blue",
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Q-Learning Performance on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("\nClose the trajectory plot to view the next plot.")
    plt.show()


def generate_td_error_plot(td_errors):
    # Plots the average absolute TD error for each trajectory
    # A declining TD error generally indicates that the learned Q-values are becoming more stable

    average_td_error = running_mean(td_errors)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Average Absolute TD Error", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    plt.plot(
        np.arange(len(td_errors)),
        td_errors,
        linewidth=1,
        label="Trajectory TD error"
    )

    if len(average_td_error) > 0:
        average_x = np.arange(
            49,
            49 + len(average_td_error)
        )

        plt.plot(
            average_x,
            average_td_error,
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Q-Learning TD Error on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("Close the TD-error plot to view the next plot.")
    plt.show()


def generate_success_rate_plot(successes,window_size=50):
    # Plots the rolling success rate over the most recent trajectories

    success_percentages = successes.astype(float) * 100
    rolling_success_rate = running_mean(
        success_percentages,
        window_size
    )

    plt.figure(figsize=(15, 7))
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xlabel("Training Trajectories", fontsize=12)

    if len(rolling_success_rate) > 0:
        success_x = np.arange(
            window_size - 1,
            window_size - 1 + len(rolling_success_rate)
        )

        plt.plot(
            success_x,
            rolling_success_rate,
            linewidth=3,
            label=f"{window_size}-trajectory success rate"
        )

    plt.ylim(-5, 105)
    plt.title("Q-Learning Success Rate on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("Run complete. Close the plot window to exit.")
    plt.show()


def running_mean(values, window_size=50):
    if len(values) < window_size:
        return np.array([])

    kernel = np.ones(window_size) / window_size

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )


if __name__ == "__main__":
    main()