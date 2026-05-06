import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ENV_NAME = "CartPole-v1"
OBSERVATION_SIZE = 4


def agent(observation, weights):
    """Return Gymnasium action 1 for right, 0 for left."""
    return 1 if np.dot(observation, weights) >= 0 else 0


def run_episode(weights, max_steps=200, seed=None):
    env = gym.make(ENV_NAME)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent(observation, weights)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return total_reward


def random_weights(rng):
    return rng.uniform(low=-1.0, high=1.0, size=OBSERVATION_SIZE)


def random_search(
    num_samples=10000,
    target_score=200,
    max_steps=200,
    rng=None,
    stop_when_target_reached=False,
):
    if rng is None:
        rng = np.random.default_rng()

    best_score = -1.0
    best_weights = None
    episodes_until_target = None

    for episode_index in range(1, num_samples + 1):
        weights = random_weights(rng)
        score = run_episode(weights, max_steps=max_steps)

        if score > best_score:
            best_score = score
            best_weights = weights

        if score >= target_score and episodes_until_target is None:
            episodes_until_target = episode_index
            if stop_when_target_reached:
                break

    return best_weights, best_score, episodes_until_target


def repeated_random_search(
    num_searches=1000,
    num_samples=10000,
    target_score=200,
    max_steps=200,
    seed=0,
):
    rng = np.random.default_rng(seed)
    episodes_needed = []
    failures = 0

    for search_index in range(1, num_searches + 1):
        _, best_score, count = random_search(
            num_samples=num_samples,
            target_score=target_score,
            max_steps=max_steps,
            rng=rng,
            stop_when_target_reached=True,
        )

        if count is None:
            failures += 1
            count = num_samples

        episodes_needed.append(count)
        print(
            "Search [%d/%d], best_score=%.0f, episodes_until_%d=%d"
            % (search_index, num_searches, best_score, target_score, count)
        )

    return np.array(episodes_needed), failures


def plot_histogram(episodes_needed, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(episodes_needed, bins=40, edgecolor="black")
    plt.xlabel("Episodes required until score 200")
    plt.ylabel("Number of searches")
    plt.title("CartPole random search")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def parse_args():
    parser = argparse.ArgumentParser(description="CartPole random search assignment")
    parser.add_argument("--num-searches", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--target-score", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    best_weights, best_score, first_success = random_search(
        num_samples=args.num_samples,
        target_score=args.target_score,
        max_steps=args.max_steps,
        rng=rng,
        stop_when_target_reached=False,
    )

    episodes_needed, failures = repeated_random_search(
        num_searches=args.num_searches,
        num_samples=args.num_samples,
        target_score=args.target_score,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    average_episodes = episodes_needed.mean()

    histogram_path = output_dir / "cartpole_random_search_histogram.png"
    plot_histogram(episodes_needed, histogram_path)

    summary = (
        "CartPole random search results\n"
        "Environment: %s\n"
        "Searches: %d\n"
        "Max random weight samples per search: %d\n"
        "Target score: %d\n"
        "Single full random-search training samples: %d\n"
        "Best score in full training: %.0f\n"
        "First successful sample in full training: %s\n"
        "Average episodes required until score %d: %.2f\n"
        "Failed searches: %d\n"
        "Best weights from full training: %s\n"
        "Histogram: %s\n"
        % (
            ENV_NAME,
            args.num_searches,
            args.num_samples,
            args.target_score,
            args.num_samples,
            best_score,
            str(first_success),
            args.target_score,
            average_episodes,
            failures,
            np.array2string(best_weights, precision=4),
            histogram_path,
        )
    )
    summary_path = output_dir / "cartpole_summary.txt"
    summary_path.write_text(summary)

    print()
    print(summary)


if __name__ == "__main__":
    main()
