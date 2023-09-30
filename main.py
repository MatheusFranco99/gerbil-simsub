"""Performs optimization by requested analysis or method"""

from typing import Dict
import numpy as np

from simulator import ( AVG_LATENCY_METRIC, CONNECTIONS_METRIC, D_GOSSIP_METRIC, D_METRIC,
                        GRAFT_METRIC, IHAVE_METRIC, IWANT_METRIC,
                        LATENCY_STD_DEV_METRIC, NODES_METRIC, NUM_MESSAGES_METRIC, NUM_PUBLISH_METRIC,
                        Simulator)
from colors import YELLOW,END

from optimization_methods import exhaustive_analysis, stochastic_gradient_descent
from simulator_conf import SimulationConf


def cost_function(metrics: Dict[str,float], verbose = True, log: callable = print) -> float:
    """Returns a single value associated to the metrics input.
    > The smaller the cost function value, the better the parameters.
    This is a crucial function and it can be further explored.
    Basically, one can implement a:
    - Weighted sum function.
    - Geometric mean or product function.
    - Max-Min Normalization with weighted sum function.
    - many more forms.
    We go with the basic weighted sum with pre-defined weights.
    """

    # Number of messages published
    m = metrics[NUM_PUBLISH_METRIC]

    # Number of nodes
    n = metrics[NODES_METRIC]

    # Mesh parameters
    D = metrics[D_METRIC]
    D_gossip = metrics[D_GOSSIP_METRIC]

    graft_cost          = (0.1) * metrics[GRAFT_METRIC] / (n*D)                                     # Graft per number of nodes and D
    ihave_cost          = (1) * metrics[IHAVE_METRIC] / (m*D_gossip)                                # IHAVE per messages and D_gossip
    iwant_cost          = (1) * (metrics[IWANT_METRIC] if IWANT_METRIC in metrics else 0) / (m*n*D) # IWANT per messages published and number of nodes
    num_messages_cost   = (1) * metrics[NUM_MESSAGES_METRIC] / (n*m*D)                              # Number of messages exchanged per number of nodes, messages and D
    connections_cost    = (1) * metrics[CONNECTIONS_METRIC] / (n*D)                                 # Number of connections per number of nodes and D
    avg_latency_cost    = (1/200) * metrics[AVG_LATENCY_METRIC]                                     # Average latency
    latency_stddev_cost = (0.1*1/200) * metrics[LATENCY_STD_DEV_METRIC]                             # Latency standard deviation

    cost = graft_cost + ihave_cost + iwant_cost + num_messages_cost + connections_cost + avg_latency_cost + latency_stddev_cost

    if verbose:
        log("Costs:")
        log(f"\t\t{graft_cost=}")
        log(f"\t\t{ihave_cost=}")
        log(f"\t\t{iwant_cost=}")
        log(f"\t\t{num_messages_cost=}")
        log(f"\t\t{connections_cost=}")
        log(f"\t\t{avg_latency_cost=}")
        log(f"\t\t{latency_stddev_cost=}")
        log(f"\tTotal: {YELLOW}{cost}{END}")

    return cost


def run_optimization_method(simulator: Simulator, x: np.array) -> None:
    """Runs optimization method"""

    func = lambda x: cost_function(simulator.extract_metrics(simulator.run_simulation(x))[0],verbose=False)
    best_x, cost, x_tested, costs = stochastic_gradient_descent(func, x, indices_to_update = [0,3], steps = [1,1,1,0.1,1,1,1,0.1],
                                                                min_bounds=[1,1,1,0.1,1,1,1,0.1],max_bounds=[15,5,16,1,6,3,6,0.25],
                                                                num_iterations = 3000)
    print(f"{len(x_tested)} simulations performed. Max cost: {max(costs)}. Min cost: {min(costs)}")
    print(f"Best result {best_x}")
    print(f"Best cost {cost}")

def run_single_instance(simulator: Simulator, x: np.array) -> None:
    """Runs a single instance"""
    result = simulator.run_simulation(x, repeat = 5)
    metrics = simulator.extract_metrics(result)
    cost = cost_function(metrics[0])
    print(metrics)
    print(cost)

def get_vector(D, D_low, D_high,
               heartbeat, history, gossip_window,
               D_gossip, gossip_factor,
               nodes, sources, messages, message_interval,
               min_latency, max_latency,
               init_delay, linger) -> np.array:
    return [D, D_low, D_high,
            heartbeat, history, gossip_window,
            D_gossip, gossip_factor,
            nodes, sources, messages, message_interval,
            min_latency, max_latency,
            init_delay, linger]
if __name__ == "__main__":

    simulator_instance = Simulator(filename="simulator_data_gerbil.pkl",use_gerbil=True)

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
                                    messages = 40,
                                    message_interval = 0.12,
                                    min_latency = 0.1,
                                    max_latency = 0.3,
                                    init_delay = 1,
                                    linger = 1)
    x_vector = simulation_cfg.to_vector()

    # Run single simulation instance, printing 
    # run_single_instance(simulator_instance,x_vector)

    # Runs optimization method
    # run_optimization_method(simulator_instance,x_vector)

    # Runs exhaustive analysis
    exhaustive_analysis(simulator_instance,x_vector,cost_function,{
                        #  0:{'name':'D','values':[2,3,4,5,6,7,8,9]},
                         3:{'name':'heartbeat','values':[100,300,500,700]},
                        #  6:{'name':'D_gossip','values':[2, 4, 6, 8]},
                        #  7:{'name':'gossip_factor','values':[0.1, 0.25, 0.5, 0.75]},
                        #  4:{'name':'history','values':[3, 6, 9, 12]},
                        #  5:{'name':'gossip window','values':[1, 3, 6]},
                        #  10:{'name':'Messages','values':[5,10,20,50,100]},
                        #  11:{'name':'Message Interval','values':[10, 50, 100, 500]},
                         }, repeat = 5, verbose=True)
