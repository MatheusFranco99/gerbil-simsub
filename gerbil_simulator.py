"""Gerbil simulator API given by the run_simulation_gerbil function"""

import os
import subprocess
import numpy as np

from simulator_conf import SimulationConf


# Temporary file for writing simulation specification
SIMULATION_FILE = "simulation_test.ss"

def simulation_code(x_vector: np.ndarray) -> str:
    """Returns the simulation code for a test with the input parameters"""

    D, D_low, D_high, heartbeat, history, gossip_window, D_gossip, gossip_factor, nodes, sources, messages, message_interval, min_latency, max_latency, init_delay, linger = map(int, x_vector)

    gossip_factor = float(x_vector[7])
    heartbeat = float(x_vector[3])
    message_interval = float(x_vector[11])
    min_latency = float(x_vector[12])
    max_latency = float(x_vector[13])

    ## Parameter sanitization
    D = max(1,int(D))
    D_low = max(1,int(D_low))
    D_high = max(1,int(D_high))
    heartbeat = min(1,max(0.1,heartbeat))
    history = max(1,int(history))
    gossip_window = min(                            # Minimum between:      
                        history,                    # - history size
                        max(1,int(gossip_window))     # - gossip window input size
                        )
    D_gossip = max(1,int(D_gossip))
    gossip_factor = min(1,max(0,gossip_factor))
    px = D

    return f"""#!/usr/bin/env gxi

(import "simsub/scripts")

(simple-gossipsub/v1.1-simulation
    D: {D}
    D-low: {D_low}
    D-high: {D_high}
    heartbeat: {heartbeat}
    initial-heartbeat-delay: {0.1}
    history: {history}
    gossip-window: {gossip_window}
    D-gossip: {D_gossip}
    gossip-factor: {gossip_factor}
    flood-publish: {"#f"}
    px: {px}
    nodes: {nodes}
    sources: {sources}
    messages: {messages}
    message-interval: {message_interval}
    init-delay: {init_delay}
    connect: {D}
    linger: {linger}
    trace: void
    min-latency: {min_latency}
    max-latency: {max_latency}
    )"""


def run_simulation_gerbil(x_vector: np.ndarray) -> str:
    """Run the simulator in Gerbil"""

    # Create simulation with given parameters
    simulation_code_text = simulation_code(x_vector)

    with open(SIMULATION_FILE,"w",encoding='utf-8') as f:
        f.write(simulation_code_text)

    # Allows file to be executed
    os.system(f'chmod 0770 {SIMULATION_FILE}')

    # Runs the simulation
    command = [f'./{SIMULATION_FILE}']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout

    except subprocess.CalledProcessError as exception:
        print(f"Error: {exception}")
        raise exception


if __name__ == "__main__":

    simulation_cfg = SimulationConf(D = 6,
                                    D_low = 4,
                                    D_high = 12,
                                    heartbeat = 0.7,
                                    history = 6,
                                    gossip_window = 4,
                                    D_gossip = 6,
                                    gossip_factor = 0.25,
                                    nodes = 72,
                                    sources = 72,
                                    messages = 8,
                                    message_interval = 0.12,
                                    min_latency = 0.002,
                                    max_latency = 0.007,
                                    init_delay = 1,
                                    linger = 1)
    x_vector = simulation_cfg.to_vector()

    print(run_simulation_gerbil(x_vector))
