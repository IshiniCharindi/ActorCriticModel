from dataclasses import dataclass
import numpy as np
from tlcs.agent import ACAgent
from tlcs.env import Environment, EnvStats

@dataclass
class Record:
    state: np.ndarray
    action: int
    reward: float
    log_prob: float
    entropy: float

def run_episode(env: Environment, agent: ACAgent, seed: int) -> tuple[list[Record], list[EnvStats]]:
    env.generate_routefile(seed=seed)
    previous_total_wait = 0.0
    history: list[Record] = []
    env_stats: list[EnvStats] = []
    env.activate()
    while not env.is_over():
        state = env.get_state()
        action, _ = agent.model.action_value(state)
        action = int(action)
        action_stats = env.execute(action)
        env_stats.extend(action_stats)
        current_total_wait = env.get_cumulated_waiting_time()
        raw = previous_total_wait - current_total_wait
        reward = float(raw) / 100.0
        previous_total_wait = current_total_wait
        record = Record(state=state, action=action, reward=reward, log_prob=0.0, entropy=0.0)
        history.append(record)
        agent.remember(state, action, reward, env.get_state(), env.is_over())
        if len(agent.memory) >= agent.n_step_size:
            agent.learn()
    final_state = env.get_state()
    dummy_record = Record(state=final_state, action=-1, reward=0.0, log_prob=0.0, entropy=0.0)
    history.append(dummy_record)
    env.deactivate()
    return history, env_stats