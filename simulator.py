"""Simulator abstraction of the GossipSub p2p parameters"""

import os
import pickle
import statistics
from typing import Dict, List
import numpy as np

from colors import END, YELLOW
from gerbil_simulator import run_simulation_gerbil
from own_simulator import run_own_simulation

# Constants for metrics keys
GRAFT_METRIC = 'graft'
IHAVE_METRIC = 'ihave'
IWANT_METRIC = 'iwant'
NUM_MESSAGES_METRIC = 'Messages'
CONNECTIONS_METRIC = 'Connections'
NODES_METRIC = 'Number of Nodes'
D_METRIC = 'D'
D_GOSSIP_METRIC = 'D_gossip'
AVG_LATENCY_METRIC = 'Average Latency'
LATENCY_STD_DEV_METRIC = 'Latency Stdev'
NUM_PUBLISH_METRIC = 'Number of Messages Published'
DELIVERY_METRIC = 'Delivery Success'
TOTAL_MESSAGES_METRIC = "Total messages"


class Simulator:
    """Class for simulating gossipsub in Gerbil and managing the results"""
    def __init__(self, filename: str = "simulator_data.pkl", use_gerbil = False):

        # Object file name
        self.obj_file = filename
        print(f"Using file: {YELLOW}{self.obj_file}{END}")
        self.use_gerbil = use_gerbil

        # Data with simulation results
        self.data = {}

        # Load data from stored file if it exists
        if os.path.isfile(self.obj_file):
            with open(self.obj_file,'rb') as f:
                self.data = pickle.load(f)

    def store(self):
        """Store simulation results"""
        with open(self.obj_file,'wb') as f:
            pickle.dump(self.data,f)

    def run_simulation(self,x_vector: np.array, repeat: int | None = None) -> str:
        """Given a configuration, runs the simulation and returns the output"""

        if repeat is None:
            repeat = 1

        # Check simulation cache
        x_vector_signature = str(x_vector)
        if x_vector_signature in self.data and len(self.data[x_vector_signature]) >= repeat:
            return self.data[x_vector_signature]

        for _ in range(repeat):

            result = ""
            if self.use_gerbil:
                result = run_simulation_gerbil(x_vector)
            else:
                result = run_own_simulation(x_vector)

            if x_vector_signature not in self.data:
                self.data[x_vector_signature] = []
            self.data[x_vector_signature] += [result]
            self.store()
        return self.data[x_vector_signature]

    def get_optimization_function(self, cost_function: callable, repeat: int | None = None) -> callable:
        """Get function to be used in optimization: x(vector) -> cost(int)"""

        return lambda x: cost_function(self.extract_metrics(self.run_simulation(x, repeat = repeat))[0],verbose=False)

    def extract_metrics(self, outputs: List[str] | str) -> Dict[str,float]:
        """get metrics from simulation output"""

        if isinstance(outputs,str):
            outputs = [outputs]

        metrics_lst = []
        for output in outputs:
            metrics = {}
            latency_histogram = {}
            in_latency_histogram = False

            for line in output.split("\n"):
                if not in_latency_histogram:
                    if "gossipsub.graft" in line:
                        metrics[GRAFT_METRIC] = int(line.split("gossipsub.graft: ")[1])
                    if "gossipsub.ihave" in line:
                        metrics[IHAVE_METRIC] = int(line.split("gossipsub.ihave: ")[1])
                    if "gossipsub.iwant" in line:
                        metrics[IWANT_METRIC] = int(line.split("gossipsub.iwant: ")[1])
                    if "pubsub.message" in line:
                        metrics[NUM_MESSAGES_METRIC] = int(line.split("pubsub.message: ")[1])
                    if "pubsub.connect" in line:
                        metrics[CONNECTIONS_METRIC] = int(line.split("pubsub.connect: ")[1])
                    if "delivery latency histogram" in line:
                        in_latency_histogram = True
                    if "pubsub.publish" in line:
                        metrics[NUM_PUBLISH_METRIC] = int(line.split("!!pubsub.publish: ")[1])
                    if "nodes: " in line:
                        metrics[NODES_METRIC] = int(line.split("nodes: ")[1])
                    if "D: " in line:
                        metrics[D_METRIC] = int(line.split("D: ")[1])
                    if "D-gossip: " in line:
                        metrics[D_GOSSIP_METRIC] = int(line.split("D-gossip: ")[1])

                else:
                    if "-" in line and "ms" in line:
                        # Gets latency and frequency
                        initial_period_latency = int(line.split()[0].split("-")[0])
                        final_period_latency = int(line.split()[0].split("-")[1].split("ms")[0])
                        mean_latency = (initial_period_latency+final_period_latency)/2

                        absolute_frequency = int(line.split()[1])

                        latency_histogram[mean_latency] = absolute_frequency

            num_latencies = sum(latency_histogram.values())
            if self.use_gerbil:
                metrics[DELIVERY_METRIC] = num_latencies / (metrics[NUM_PUBLISH_METRIC] * metrics[NODES_METRIC] - metrics[NUM_PUBLISH_METRIC])
            else:
                metrics[DELIVERY_METRIC] = num_latencies / (metrics[NUM_PUBLISH_METRIC] * (metrics[NODES_METRIC]-1))

            if self.use_gerbil:
                metrics[TOTAL_MESSAGES_METRIC] = metrics[NUM_MESSAGES_METRIC]
            else:
                metrics[TOTAL_MESSAGES_METRIC] = metrics[NUM_MESSAGES_METRIC] + metrics[IHAVE_METRIC] + metrics[IWANT_METRIC]

            # Latency mean and standard deviation calculation
            latencies = np.array(list(latency_histogram.keys()))
            frequencies = np.array(list(latency_histogram.values()))
            average_latency = np.average(latencies, weights = frequencies)
            weighted_variance = np.average((latencies - average_latency) ** 2, weights = frequencies)
            standard_deviation = np.sqrt(weighted_variance)

            metrics[AVG_LATENCY_METRIC] = average_latency
            metrics[LATENCY_STD_DEV_METRIC] = standard_deviation

            metrics_lst.append(metrics)

        metrics_keys = metrics_lst[0].keys()
        average_metric = {key: [] for key in metrics_keys}
        stddev_metric = {key: [] for key in metrics_keys}
        for metric in metrics_lst:
            for key in metrics_keys:
                average_metric[key] += [metric[key]]
                stddev_metric[key] += [metric[key]]

        for key in metrics_keys:
            average_metric[key] = statistics.mean(average_metric[key])
            stddev_metric[key] = statistics.stdev(stddev_metric[key])


        return average_metric, stddev_metric
