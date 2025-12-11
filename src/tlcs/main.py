from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TypedDict
import numpy as np
import tensorflow as tf
import random
import glob
from tlcs.agent import ACAgent
from tlcs.constants import TESTING_SETTINGS_FILE, TRAINING_SETTINGS_FILE,STATE_SIZE, NUM_ACTIONS
from tlcs.env import Environment, EnvStats
from tlcs.episode import Record, run_episode
from tlcs.logger import get_logger
from tlcs.plots import save_data_and_plot
from tlcs.settings import load_testing_settings, load_training_settings
logger = get_logger(__name__)
class TrainingStats(TypedDict):
    sum_neg_reward: list[float]
    cumulative_wait: list[int]
    avg_queue_length: list[float]
class TestingStats(TypedDict):
    reward: list[float]
    queue_length: list[int]
def update_training_stats(
        episode_history: list[Record],
        env_stats: list[EnvStats],
        max_steps: int,
        training_stats: TrainingStats,
) -> TrainingStats:
    effective_history = [r for r in episode_history if r.action >= 0]
    sum_neg_reward = sum(record.reward for record in effective_history if record.reward < 0)
    training_stats["sum_neg_reward"].append(sum_neg_reward)
    sum_queue_length = sum(stats.queue_length for stats in env_stats)
    avg_queue_length = round(sum_queue_length / max_steps, 1)
    training_stats["avg_queue_length"].append(avg_queue_length)
    training_stats["cumulative_wait"].append(sum_queue_length)
    return training_stats
def training_session(settings_file: Path, out_path: Path) -> None:
    settings = load_training_settings(settings_file)
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    agent = ACAgent(STATE_SIZE, NUM_ACTIONS, 0, settings.n_step_size, settings.gamma, settings.learning_rate, settings.beta_entropy, settings.beta_entropy)
    timestamp_start = datetime.now()
    training_stats = {
        "sum_neg_reward": [],
        "cumulative_wait": [],
        "avg_queue_length": [],
    }
    out_path.mkdir(parents=True, exist_ok=True)
    episode = 0
    logger.info("Starting unlimited training... Press CTRL+C to stop.")
    try:
        while True:
            episode += 1
            logger.info(f"Episode {episode}")
            env = Environment(
                settings.n_cars_generated,
                settings.max_steps,
                settings.yellow_duration,
                settings.green_duration,
                settings.turn_chance,
                settings.sumocfg_file,
                settings.gui,
            )
            episode_history, env_stats = run_episode(env, agent, episode)
            training_stats = update_training_stats(
                episode_history,
                env_stats,
                settings.max_steps,
                training_stats,
            )
            logger.info(f"\tReward: {training_stats['sum_neg_reward'][-1]}")
            logger.info(f"\tCumulative wait: {training_stats['cumulative_wait'][-1]}")
            logger.info(f"\tAvg queue: {training_stats['avg_queue_length'][-1]}")
    except KeyboardInterrupt:
        logger.info("Training manually stopped! Saving model...")
        agent.save_agent(out_path, 'tlcs', 'AC', 'session', episode)
    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    copyfile(settings_file, out_path / TRAINING_SETTINGS_FILE)
    save_data_and_plot(training_stats["sum_neg_reward"], "reward", "Episode", "Cumulative negative reward", out_path)
    save_data_and_plot(training_stats["cumulative_wait"], "delay", "Episode", "Cumulative delay (s)", out_path)
    save_data_and_plot(training_stats["avg_queue_length"], "queue", "Episode", "Average queue length (vehicles)", out_path)
def testing_session(settings_file: Path, model_path: Path, out_path: Path) -> None:
    settings = load_testing_settings(settings_file)
    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)
    agent = ACAgent(STATE_SIZE, NUM_ACTIONS, 0, settings.n_step_size, settings.gamma, settings.learning_rate, settings.beta_entropy, settings.beta_entropy)
    weight_files = glob.glob(str(model_path / "Agent0.weights.h5"))
    if not weight_files:
        raise FileNotFoundError("No model weights found in model_path")
    agent.load_agent(model_path, 'tlcs', 'AC', 'session', 0, best=True)
    tot_episodes = settings.total_episodes
    stats = {
        "reward": [],
        "queue_length": [],
    }
    for episode in range(tot_episodes):
        logger.info(f"[TEST] Episode {episode + 1} of {tot_episodes}")
        env = Environment(
            settings.n_cars_generated,
            settings.max_steps,
            settings.yellow_duration,
            settings.green_duration,
            settings.turn_chance,
            settings.sumocfg_file,
            settings.gui,
        )
        history, env_stats = run_episode(env, agent, episode)
        episode_reward = sum(r.reward for r in history if r.action >= 0)
        total_queue = sum(s.queue_length for s in env_stats)
        stats["reward"].append(episode_reward)
        stats["queue_length"].append(total_queue)
        logger.info(f"\tReward: {episode_reward}")
        logger.info(f"\tQueue length: {total_queue}")
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Testing finished. Saving results to {out_path}")
    copyfile(settings_file, out_path / TESTING_SETTINGS_FILE)
    save_data_and_plot(stats["reward"], "reward", "Episode", "Reward", out_path)
    save_data_and_plot(stats["queue_length"], "queue", "Episode", "Queue length", out_path)