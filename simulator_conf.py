"""Simulation configuration"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationConf:
    """Simulation configuration"""
    D: int
    D_low: int
    D_high: int
    heartbeat: float
    history: int
    gossip_window: int
    D_gossip: int
    gossip_factor: float
    nodes: int
    sources: int
    messages: int
    message_interval: float
    min_latency: float
    max_latency: float
    init_delay: float
    linger: int

    def to_vector(self) -> np.array:
        return [self.D, self.D_low, self.D_high,
                self.heartbeat, self.history,self.gossip_window,
                self.D_gossip, self.gossip_factor,
                self.nodes, self.sources,
                self.messages, self.message_interval,
                self.min_latency, self.max_latency,
                self.init_delay, self.linger
                ]