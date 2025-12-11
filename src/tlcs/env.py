from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from sumolib import checkBinary
from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    CELLS_PER_LANE_GROUP,
    INCOMING_EDGES,
    LANE_DISTANCE_TO_CELL,
    LANE_ID_TO_GROUP,
    ROAD_MAX_LENGTH,
    STATE_SIZE,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)
from tlcs.generator import generate_routefile

@dataclass
class EnvStats:
    queue_length: int

class Environment:
    def __init__(
            self,
            n_cars_generated,
            max_steps,
            yellow_duration,
            green_duration,
            turn_chance,
            sumocfg_file,
            gui,
    ):
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.sumocfg_file = Path(sumocfg_file)
        self.gui = gui
        self.step = 0

    def build_sumo_cmd(self):
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")
        if not self.sumocfg_file.exists():
            raise FileNotFoundError(f"SUMO config not found at '{self.sumocfg_file}'")
        return [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]

    def activate(self):
        sumo_cmd = self.build_sumo_cmd()
        traci.start(sumo_cmd)

    def deactivate(self):
        traci.close()

    def is_over(self):
        return self.step >= self.max_steps

    def generate_routefile(self, seed):
        generate_routefile(seed, self.n_cars_generated, self.max_steps, self.turn_chance)

    def _get_lane_cell(self, lane_pos):
        lane_pos = ROAD_MAX_LENGTH - lane_pos
        lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))
        for distance, cell in LANE_DISTANCE_TO_CELL.items():
            if lane_pos <= distance:
                return cell
        raise RuntimeError("Error while getting lane cell.")

    def get_state(self):
        state = np.zeros(STATE_SIZE, dtype=float)
        for car_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_group = LANE_ID_TO_GROUP.get(lane_id)
            if lane_group is None:
                continue
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_cell = self._get_lane_cell(lane_pos)
            car_position = lane_group * CELLS_PER_LANE_GROUP + lane_cell
            state[car_position] += 1.0
        state = np.clip(state / 50.0, 0.0, 1.0)
        return state[np.newaxis, :]

    def get_cumulated_waiting_time(self):
        waiting_times = 0.0
        for car_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id not in INCOMING_EDGES:
                continue
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            waiting_times += wait_time
        return waiting_times

    def _set_yellow_phase(self, green_phase_code):
        yellow_phase_code = TL_GREEN_TO_YELLOW[green_phase_code]
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_code)

    def _set_green_phase(self, green_phase_code):
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, green_phase_code)

    def _simulate(self, duration):
        stats = []
        steps_todo = min(duration, self.max_steps - self.step)
        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            queue_length = self.get_queue_length()
            stats.append(EnvStats(queue_length=queue_length))
        return stats

    def execute(self, action):
        next_green_phase = ACTION_TO_TL_PHASE[action]
        current_green_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
        stats = []
        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats_yellow = self._simulate(self.yellow_duration)
            stats.extend(stats_yellow)
        if self.is_over():
            return stats
        self._set_green_phase(next_green_phase)
        stats_green = self._simulate(self.green_duration)
        stats.extend(stats_green)
        return stats

    def get_queue_length(self):
        halt_n = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_s = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_e = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_w = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_n + halt_s + halt_e + halt_w)