#!/usr/bin/env gxi

(import :gerbil/gambit
        :std/format
        :std/iter
        "../scheduler"
        "../scripts"
        "../episub")

(def (run-simulations nodes sources messages)
  (def rng
    (let (rng (make-random-source))
      (random-source-randomize! rng)
      rng))
  (def rng-state
    (random-source-state-ref rng))
  (def (run-it what run)
    (printf "+++ ~a~n" what)
    (##gc)
    (run))
  (def (reset!)
    (virtual-time-scheduler-reset!))

  (printf ">>> Running simulations with nodes: ~a, sources: ~a, messages: ~a, rng-state: ~a~n" nodes sources messages rng-state)
  (enable-virtual-time-scheduler!)
  (run-it 'gossipsub/v1.0
          (lambda ()
            (simple-gossipsub/v1.0-simulation
             nodes: nodes sources: sources messages: messages
             rng: rng
             init-delay: 10
             trace: void)))
  (reset!)
  (run-it 'gossipsub/v1.1
          (lambda ()
            (simple-gossipsub/v1.1-simulation
             nodes: nodes sources: sources messages: messages
             rng: rng
             init-delay: 10
             trace: void)))
  (for (strategy '(order-avg order-median latency-avg latency-median latency-p90))
    (reset!)
    (run-it (format "episub/~a" strategy)
            (lambda ()
              (simple-episub-simulation
               params: (make-overlay/v1.2 choke-strategy: strategy)
               nodes: nodes sources: sources messages: messages
               rng: rng
               init-delay: 10
               trace: void)))))

(for* ((nodes '(100))
       (sources '(5 10 20))
       (messages '(60 120 300)))
  (run-simulations nodes sources messages))
