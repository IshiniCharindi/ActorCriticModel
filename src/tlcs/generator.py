import numpy as np
from tlcs.constants import ROUTES_FILE, ROUTES_FILE_HEADER, STRAIGHT_ROUTES, TURN_ROUTES

def _map_to_interval(values, new_min, new_max):
    old_min = float(values.min())
    old_max = float(values.max())
    return np.interp(values, (old_min, old_max), (new_min, new_max))

def _get_car_row(route_id, car_i, step):
    return f' <vehicle id="{route_id}_{car_i}" type="standard_car" route="{route_id}" depart="{step}" departLane="random" departSpeed="10" />'

def generate_routefile(seed, n_cars_generated, max_steps, turn_chance):
    rng = np.random.default_rng(seed)
    timings = np.sort(rng.weibull(2.0, size=n_cars_generated))
    generated_steps = _map_to_interval(timings, new_min=0, new_max=max_steps)
    depart_steps = np.rint(generated_steps).astype(int)
    ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ROUTES_FILE, "w", encoding="utf-8") as routes_file:
        routes_file.write(ROUTES_FILE_HEADER + "\n")
        for car_i, step in enumerate(depart_steps):
            routes_selected = TURN_ROUTES if rng.random() < turn_chance else STRAIGHT_ROUTES
            route_id = rng.choice(routes_selected)
            car_row = _get_car_row(route_id, car_i, step)
            routes_file.write(car_row + "\n")
        routes_file.write("</routes>")