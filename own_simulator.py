""" Own python simulator of the GossipSub protocol 
    - Class Simulator with a method "run" that returns the simulation result
    - function "run_own_simulation" to server as external caller
"""

from dataclasses import dataclass
import heapq
from enum import Enum
import statistics
from typing import Dict, List
import random

import numpy as np


class EventType(Enum):
    """Type of event for the simulator to handle"""
    NODE_SEND_MESSAGE = 1
    MESSAGE = 2
    CONTROL_MESSAGE = 3
    IWANT_MESSAGE = 4
    HEARTBEAT = 5

@dataclass
class EventObject:
    """General Event Object"""
    timestamp: int
    sender: int
    message_id: int

@dataclass
class EventNodeSendMessageObject(EventObject):
    """Event to trigger a node to publish a message"""

@dataclass
class EventMessageObject(EventObject):
    """Event that represents a full-message message"""

@dataclass
class EventControlMessageObject(EventObject):
    """Gossip Message"""
    ihave: List[int]
    graft: List[int]
    prune: List[int]

@dataclass
class EventIwantMessageObject(EventObject):
    """IWANT message"""

@dataclass
class EventHeartbeatObject(EventObject):
    """Heartbeat trigger for simulator to call nodes gossip function"""




class Event:
    """Event to be handled by the simulator"""

    def __init__(self, event_type: EventType, event_obj: EventObject, receiver: int | List[int]):
        self.event_type = event_type
        self.event_obj = event_obj
        self.receiver = receiver # Node or nodes that should process the event

    def __lt__(self, other) -> bool:
        return self.event_obj.timestamp < other.event_obj.timestamp

    def __repr__(self) -> str:
        return f"{self.event_type} - {self.event_obj} - {self.receiver}"


class Node:

    def __init__(self, peer_id: int, D: int, D_low: int, D_high: int,
                 heartbeat: int, mcache_len: int, mcache_gossip: int,
                 D_gossip: int, gossip_factor: float,
                 min_latency: int, max_latency: int):

        self.id = peer_id
        self.D = D
        self.D_low = D_low
        self.D_high = D_high
        self.heartbeat = heartbeat
        self.mcache_len = mcache_len
        self.mcache_gossip = mcache_gossip
        self.D_gossip = D_gossip
        self.gossip_factor = gossip_factor
        self.min_latency = min_latency
        self.max_latency = max_latency

        # Node's mesh
        self.mesh = set()
        # Node's message cache
        self.mcache = [set() for i in range(mcache_len)]
        # Node's seen dictionary with no seen_ttl. Dictionary of type MessageID (int) -> timestamp
        self.seen = {}

        # Stores published messages ids
        self.own_msgs = set()

    def __lt__(self, other) -> bool:
        return len(self.mesh) < len(other.mesh)

    def can_connect(self) -> bool:
        """Check if mesh is not full. Used for pre-simulation connection"""
        return len(self.mesh) < self.D_high

    def establish_connection(self, peer: int) -> None:
        """Create full-message link. Used for pre-simulation connection"""
        self.mesh.add(peer)

    def connect(self, nodes_lst: List) -> None:
        """Fills node's mesh. Used for pre-simulation connection"""

        # Filter nodes for which mesh are not full
        nodes = [n for n in nodes_lst if len(n.mesh) < self.D_high]
        # Sort nodes according to mesh fulfillment
        nodes = [x for _, x in sorted(zip([len(n.mesh) for n in nodes],nodes))]

        # index for nodes list
        idx = 0

        while len(self.mesh) < self.D:
            # Take peer
            peer = nodes[idx%(len(nodes))]
            idx += 1

            # Assert it's not itself or it's already in mesh
            if peer.id == self.id or peer.id in self.mesh:
                continue

            if peer.can_connect():
                self.establish_connection(peer.id)

    def get_seen(self) -> Dict[int,int]:
        """Returns the seen dictionary"""
        return self.seen

    def get_delay(self) -> int:
        """Get random delay"""
        return random.randint(self.min_latency,self.max_latency)

    def process_event(self, event: Event, add_event_f: callable) -> None:
        """Main function to handle event"""

        if event.event_type == EventType.NODE_SEND_MESSAGE:
            self.send(event.event_obj,add_event_f)
        elif event.event_type == EventType.MESSAGE:
            self.process_message(event.event_obj,add_event_f)
        elif event.event_type == EventType.CONTROL_MESSAGE:
            self.process_control_message(event.event_obj,add_event_f)
        elif event.event_type == EventType.IWANT_MESSAGE:
            self.process_iwant_message(event.event_obj,add_event_f)
        else:
            raise TypeError(f"Event with unexpected type: {event.event_type}")

    def send(self, event_obj: EventNodeSendMessageObject, add_event_f: callable) -> None:
        """Handles Send event type. Node should publish the message.
        Thus, it sends a Message event to the peers in its mesh"""

        # Check if self is the publisher
        if self.id != event_obj.sender:
            raise ValueError(f"Requested to send message but sender is not itself.\
                             Sender {event_obj}. ID: {self.id}")

        # Add to state
        self.mcache[0].add(event_obj.message_id)
        self.seen[event_obj.message_id] = 0
        self.own_msgs.add(event_obj.message_id)

        # Sends Message
        for peer in self.mesh:
            event = Event(EventType.MESSAGE,
                          EventMessageObject(timestamp = event_obj.timestamp + self.get_delay(),
                                             sender = self.id,
                                             message_id = event_obj.message_id),
                          receiver = peer)
            add_event_f(event)

    def process_message(self, event_obj: EventMessageObject, add_event_f: callable) -> None:
        """ Handles Message event type.
            Should check if has never seen it and, if so, propagates to the peers in its mesh
        """

        # Already handled
        if event_obj.message_id in self.seen:
            return

        # Add to state
        self.seen[event_obj.message_id] = event_obj.timestamp
        self.mcache[0].add(event_obj.message_id)

        # Sends Message
        for peer in self.mesh:
            if peer == event_obj.sender:
                continue
            event = Event(EventType.MESSAGE,
                          EventMessageObject(timestamp = event_obj.timestamp + self.get_delay(),
                                             sender = self.id,
                                             message_id = event_obj.message_id),
                          receiver = peer)
            add_event_f(event)

    def process_control_message(self, event_obj: EventControlMessageObject,
                                add_event_f: callable) -> None:
        """ Handles control message. Should check if it's pruned.
            Should check if has every MessageID in ihave field and, if not, request by IWANT.
        """

        # Send IWANT in case it doesn't have a MessageID
        ihave = event_obj.ihave
        for msg_id in ihave:
            if (msg_id not in self.own_msgs) and (msg_id not in self.seen):
                event = Event(EventType.IWANT_MESSAGE,
                              EventIwantMessageObject(timestamp = event_obj.timestamp + self.get_delay(),
                                                      sender = self.id,
                                                      message_id = msg_id),
                              receiver = event_obj.sender)
                add_event_f(event)

        # Check if it's pruned
        prune = event_obj.prune
        if self.id in prune and event_obj.sender in self.mesh:
            self.mesh.remove(event_obj.sender)

        # Check if it's grafted
        graft = event_obj.graft
        if self.id in graft and event_obj.sender not in self.mesh:
            self.mesh.add(event_obj.sender)


    def has_in_cache(self, msg_id: int) -> bool:
        """ Check if has message in cache.
            Used to answer IWANT messages
        """
        for _, mcache_window in enumerate(self.mcache):
            if msg_id in mcache_window:
                return True
        return False


    def process_iwant_message(self, event_obj: EventIwantMessageObject,
                              add_event_f: callable) -> None:
        """ Handles IWANT message.
            Check if it has in cache and sends message back
            """

        if self.has_in_cache(event_obj.message_id):
            event = Event(EventType.MESSAGE,
                          EventMessageObject(event_obj.timestamp + self.get_delay(),
                                             self.id,
                                             event_obj.message_id),
                          receiver = event_obj.sender)
            add_event_f(event)

    def heartbeat_gossip(self, timestamp: int, add_event_f: callable, nodes: List[int]) -> None:
        """Periodic gossip function. Check if should graft or prune and collects IHAVE to send"""

        # Select nodes to gossip
        num_nodes_to_select = int(len(nodes) * self.gossip_factor)
        num_nodes_to_select = max(self.D_gossip, num_nodes_to_select)
        random_nodes = random.sample(nodes, num_nodes_to_select)

        prune = []
        graft = []

        # Prune if needed
        if len(self.mesh) > self.D_high:
            while len(self.mesh) != self.D:
                random_peer = random.sample(self.mesh,1)[0]
                prune += [random_peer]
                self.mesh.remove(random_peer)

        # Graft if needed
        if len(self.mesh) < self.D_low:
            while len(self.mesh) != self.D:
                random_peer = random.sample(nodes,1)[0]
                if random_peer not in self.mesh:
                    self.mesh.add(random_peer)
                    graft += [random_peer]

        # Collects MessageIDs to fill IHAVE field
        ihave = set()
        for i in range(0,self.mcache_gossip):
            for msg_id in self.mcache[i]:
                ihave.add(msg_id)

        ihave = list(ihave)

        event = Event(EventType.CONTROL_MESSAGE,
                        EventControlMessageObject(timestamp + self.get_delay(),
                                                  sender = self.id,
                                                  message_id = -2,
                                                  ihave = ihave,
                                                  graft = graft,
                                                  prune = prune),
                        receiver = random_nodes)
        add_event_f(event)

        # Update message cache
        self.mcache = [set()] + self.mcache[0:-1]


class OwnSimulator:
    """GossipSub p2p network simulator"""

    def __init__(self, nodes: int, messages_per_node: int, message_delay: int,
                 D: int, D_low: int, D_high: int,
                 heartbeat: int, mcache_len: int, mcache_gossip: int,
                 D_gossip: int, gossip_factor: float,
                 min_latency: int, max_latency: int,
                 heartbeats_after_done: int = 2):
        self.D = D
        self.D_low = D_low
        self.D_high = D_high
        self.heartbeat = heartbeat
        self.mcache_len = mcache_len
        self.mcache_gossip = mcache_gossip
        self.D_gossip = D_gossip
        self.gossip_factor = gossip_factor
        self.min_latency = min_latency
        self.max_latency = max_latency

        # Events min heap
        self.events = []

        # Counts messages processed
        self.msg_counter = {event_type: 0 for event_type in EventType}

        # Heartbeats to happen after last message is published
        self.heartbeats_after_done = heartbeats_after_done
        self.done = False

        # Counts graft and connections
        self.graft_count = 0
        self.connections = 0

        # Create Nodes
        self.nodes = {}
        for peer_id in range(1,nodes+1):
            self.nodes[peer_id] = Node(peer_id,D,D_low,D_high,
                                  heartbeat,mcache_len,mcache_gossip,
                                  D_gossip,gossip_factor,
                                  min_latency,max_latency)

        # Connect nodes (fills meshs) before running simulation
        available_nodes = {node for _, node in self.nodes.items()}
        for peer_id in range(1,nodes+1):
            self.nodes[peer_id].connect(list(available_nodes))
            available_nodes.remove(self.nodes[peer_id])

            if len(available_nodes) <= D:
                available_nodes = {node for _, node in self.nodes.items()}
        self.connections += sum([len(node.mesh) for _, node in self.nodes.items()])/2

        # Add heartbeat trigger events
        for i in range(1,200):
            self.add_event(Event(EventType.HEARTBEAT,
                                 EventHeartbeatObject(timestamp = i * heartbeat,
                                                      sender = -1,
                                                      message_id = -1),
                                 receiver =  -1))

        # Add events to trigger nodes to send messages
        self.msg_timestamp = {}
        msg_id = 0
        for peer in self.nodes:
            curr_msg_timestamp = 0
            for i in range(messages_per_node):
                msg_id += 1
                curr_msg_timestamp += message_delay
                self.add_event(Event(EventType.NODE_SEND_MESSAGE,
                                     EventNodeSendMessageObject(curr_msg_timestamp,peer,msg_id),
                                     receiver = peer))
                self.msg_timestamp[msg_id] = curr_msg_timestamp

        # Saves last message seen to know when it's done
        self.last_msg_id = len(self.nodes) * messages_per_node

    def run(self) -> None:
        """Run simulation"""

        # Keeps track of the number of heartbeats that occured after last message was published
        curr_heartbeats_after_done = 0

        while curr_heartbeats_after_done < self.heartbeats_after_done:

            # No events -> Stop
            if not self.has_event():
                print("Don't have more events")
                break

            # Takes most recent event
            event = self.get_event()

            # If not heartbeat, should be redirected to a node
            if event.event_type != EventType.HEARTBEAT:

                 # In control message, the receiver is a list of nodes
                if event.event_type == EventType.CONTROL_MESSAGE:
                    for peer_id in event.receiver:
                        self.nodes[peer_id].process_event(event,self.add_event)
                else:
                    peer_id = event.receiver
                    self.nodes[peer_id].process_event(event,self.add_event)
            else:
                # heartbeat trigger should make all nodes to gossip
                for _, peer in self.nodes.items():
                    peer.heartbeat_gossip(event.event_obj.timestamp,
                                          self.add_event,
                                          list(self.nodes.keys()))

            # Checks if last message is being published
            if event.event_type == EventType.NODE_SEND_MESSAGE:
                if event.event_obj.message_id == self.last_msg_id:
                    self.done = True

            if self.done and event.event_type == EventType.HEARTBEAT:
                curr_heartbeats_after_done += 1

            # Monitor grafts
            if event.event_type == EventType.CONTROL_MESSAGE:
                self.graft_count += len(event.event_obj.graft)


        return self.report()

    def add_event(self, event: Event) -> None:
        """Puts event in min heap"""
        heapq.heappush(self.events,event)


    def get_event(self) -> Event:
        """Get most recent event"""
        event = heapq.heappop(self.events)
        if event.event_type == EventType.CONTROL_MESSAGE:
            self.msg_counter[event.event_type] += len(event.receiver)
        else:
            self.msg_counter[event.event_type] += 1
        return event

    def has_event(self) -> bool:
        """Check if heap is not empty"""
        return len(self.events) > 0

    def report(self) -> str:
        """Creates simulation report"""

        txt = ""

        # Message event counter
        txt += str(self.msg_counter) + "\n"


        # Metrics using the nodes' seen fields
        latencies = []
        percentage_seen = []
        deliveries = 0
        for _, peer in self.nodes.items():

            seen = peer.get_seen()

            percentage_seen += [len(set(seen)) / len(self.msg_timestamp)]

            deliveries += len(set(seen))

            for msg_id in seen:
                if seen[msg_id] != 0:
                    latencies += [seen[msg_id] - self.msg_timestamp[msg_id]]

        txt += str(f"Num of latencies - {len(latencies)}") + "\n"
        txt += str(f"Latencies mean - {statistics.mean(latencies)}") + "\n"
        txt += str(f"Length of percentage seen list - {len(percentage_seen)}") + "\n"
        txt += str(f"Average percentage seen - {statistics.mean(percentage_seen)}") + "\n"

        # Divide latencies in buckets of bucket_size size
        bucket_size = 25
        max_latency = max(latencies)

        max_bucket = 0
        if max_latency % bucket_size == 0:
            max_bucket = int(max_latency)
        else:
            max_bucket = (int(max_latency/bucket_size) + 1) * bucket_size

        buckets = {v: 0 for v in range(bucket_size,max_bucket+1,bucket_size)}

        # fill buckets
        for latency in latencies:
            for v in buckets:
                if latency > v-bucket_size and latency <= v:
                    buckets[v] += 1

        txt += f"""=== simulation summary ===
nodes: {len(self.nodes)}
messages: {self.msg_counter[EventType.MESSAGE] + self.msg_counter[EventType.CONTROL_MESSAGE] + self.msg_counter[EventType.IWANT_MESSAGE]}
sources: {len(self.nodes)}
publish: {self.msg_counter[EventType.NODE_SEND_MESSAGE]}
deliver: {deliveries}
D: {self.D}
D-low: {self.D_low}
D-high: {self.D_high}
heartbeat: {self.heartbeat}
initial-heartbeat-delay: {0}
history: {self.mcache_len}
gossip-window: {self.mcache_gossip}
D-gossip: {self.D_gossip}
gossip-factor: {self.gossip_factor}
flood-publish: #f
px: 10
!!gossipsub.iwant: {int(self.msg_counter[EventType.IWANT_MESSAGE])}
!!pubsub.publish: {int(self.msg_counter[EventType.NODE_SEND_MESSAGE])}
!!gossipsub.ihave: {int(self.msg_counter[EventType.CONTROL_MESSAGE])}
!!gossipsub.graft: {int(self.graft_count)}
!!pubsub.message: {int(self.msg_counter[EventType.MESSAGE])}
!!pubsub.connect: {int(self.graft_count + self.connections)}
=== delivery latency histogram ===\n"""

        # Print latency histogram
        for v, absolute_frequency in buckets.items():
            txt += f"\t{v-bucket_size}-{v}ms\t{absolute_frequency}" + "\n"

        return txt[:-1] # remove last "\n" to avoid parsing errors


def run_own_simulation(x_vector: np.array) -> str:
    """Given configuration parameters, runs own simulator and returns the report"""
    D, D_low, D_high, heartbeat, history, gossip_window, D_gossip, gossip_factor, nodes, _, messages, message_interval, min_latency, max_latency, _, _ = map(int, x_vector)

    gossip_factor = float(x_vector[7])
    heartbeat = float(x_vector[3])
    message_interval = float(x_vector[11])
    min_latency = float(x_vector[12])
    max_latency = float(x_vector[13])

    ## Parameter sanitization
    D = max(1,int(D))
    D_low = max(1,int(D_low))
    D_high = max(1,int(D_high))
    history = max(1,int(history))
    gossip_window = min(                            # Minimum between:      
                        history,                    # - history size
                        max(1,int(gossip_window))     # - gossip window input size
                        )
    D_gossip = max(1,int(D_gossip))
    gossip_factor = min(1,max(0,gossip_factor))

    own_simulator = OwnSimulator(nodes=nodes,
                                 messages_per_node=messages,message_delay=message_interval,
                                 D=D,D_low=D_low,D_high=D_high,heartbeat=heartbeat,
                                 mcache_len=history,mcache_gossip=gossip_window,
                                 D_gossip=D_gossip,gossip_factor=gossip_factor,
                                 min_latency=min_latency,max_latency=max_latency,
                                 heartbeats_after_done=min(gossip_window,2))

    result = own_simulator.run()

    return result

if __name__ == "__main__":
    s = OwnSimulator(nodes = 72, messages_per_node = 8,
                message_delay = 125,
                D = 6, D_low = 4, D_high = 12,
                heartbeat = 700, mcache_len = 6, mcache_gossip = 4,
                D_gossip = 6, gossip_factor = 0.25,
                min_latency = 2, max_latency = 7,
                heartbeats_after_done = 10)

    output = s.run()
    print(output)
