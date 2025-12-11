from pathlib import Path

DEFAULT_SETTINGS_PATH = Path("settings")
DEFAULT_MODEL_PATH = Path("model")
TRAINING_SETTINGS_FILE = Path("training_settings.yaml")
TESTING_SETTINGS_FILE = Path("testing_settings.yaml")
MODEL_FILE = Path("trained_model.h5")
DEFAULT_TEST_FOLDER = "test"

ROUTES_FILE = Path("intersection/episode_routes.rou.xml")
ROUTES_FILE_HEADER = """<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>"""

STRAIGHT_ROUTES = ("W_E", "E_W", "N_S", "S_N")
TURN_ROUTES = ("W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E")

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7

ACTION_TO_TL_PHASE = {
    0: PHASE_NS_GREEN,
    1: PHASE_NSL_GREEN,
    2: PHASE_EW_GREEN,
    3: PHASE_EWL_GREEN,
}

TL_GREEN_TO_YELLOW = {
    PHASE_NS_GREEN: PHASE_NS_YELLOW,
    PHASE_NSL_GREEN: PHASE_NSL_YELLOW,
    PHASE_EW_GREEN: PHASE_EW_YELLOW,
    PHASE_EWL_GREEN: PHASE_EWL_YELLOW,
}

STATE_SIZE = 80  # 8 lane groups * 10 cells
NUM_ACTIONS = 4

ROAD_MAX_LENGTH = 750

LANE_DISTANCE_TO_CELL = {
    7: 0,
    14: 1,
    21: 2,
    28: 3,
    40: 4,
    60: 5,
    100: 6,
    160: 7,
    400: 8,
    750: 9,
}

CELLS_PER_LANE_GROUP = 10

INCOMING_EDGES = ("E2TL", "N2TL", "W2TL", "S2TL")

TRAFFIC_LIGHT_ID = "TL"

LANE_ID_TO_GROUP = {
    "W2TL_0": 0,
    "W2TL_1": 0,
    "W2TL_2": 0,
    "W2TL_3": 1,
    "N2TL_0": 2,
    "N2TL_1": 2,
    "N2TL_2": 2,
    "N2TL_3": 3,
    "E2TL_0": 4,
    "E2TL_1": 4,
    "E2TL_2": 4,
    "E2TL_3": 5,
    "S2TL_0": 6,
    "S2TL_1": 6,
    "S2TL_2": 6,
    "S2TL_3": 7,
}