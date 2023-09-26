#!/usr/bin/env gxi

(import "simsub/scripts")

(simple-gossipsub/v1.1-simulation
    D: 4
    D-low: 1
    D-high: 20
    heartbeat: 2
    initial-heartbeat-delay: 2
    history: 6
    gossip-window: 3
    D-gossip: 3
    gossip-factor: 0.4
    flood-publish: #f
    px: 10
    nodes: 128
    sources: 5
    messages: 10
    message-interval: 1
    init-delay: 5
    connect: 20
    linger: 10
    trace: void
    min-latency: 0.010
    max-latency: 0.150
    )