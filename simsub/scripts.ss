;; -*- Gerbil -*-
;; © vyzo
;; simple simulation scripts

(import :gerbil/gambit
        :std/iter
        :std/misc/shuffle
        (only-in :std/srfi/1 take)
        :vyzo/simsub/env
        :vyzo/simsub/proto
        :vyzo/simsub/floodsub
        :vyzo/simsub/gossipsub-base
        :vyzo/simsub/gossipsub-v1_0
        :vyzo/simsub/simulator)
(export #t)

(def (simple-gossipsub/v1.0-simulation #!key kws params: (params #f))
  (apply simple-simulation
    router: gossipsub/v1.0
    params: (or params (make-overlay/v1.0))
    (keyword-rest kws router: params:)))

(def (simple-floodsub-simulation #!key kws)
  (apply simple-simulation
    router: floodsub
    params: #f
    (keyword-rest kws router: params:)))

(def (simple-simulation nodes: (nodes 100)
                        fanout: (fanout 5)
                        messages: (messages 10)
                        message-delay: (message-delay 1)
                        connect-delay: (connect-delay 5)
                        connect: (connect 10)
                        linger: (linger 10)
                        trace: (trace displayln)
                        transcript: (transcript void)
                        single-source: (single-source #f)
                        router: router
                        params: params)
  (def traces (box []))

  (def (my-trace evt)
    (set! (box traces)
      (cons evt (unbox traces)))
    (trace evt))

  (def (my-script peers)
    (thread-sleep! connect-delay)
    (let (peers (shuffle peers))
      (let lp ((i 0))
        (when (< i messages)
          (let (dest (take (if single-source peers (shuffle peers))
                           fanout))
            (for (peer dest)
              (let (msg (cons 'msg i))
                (trace-publish! i msg)
                (send! (!!pubsub.publish peer i msg)))))
          (thread-sleep! message-delay)
          (lp (1+ i)))))
    (thread-sleep! linger))

  (def (display-summary!)
    (def publish 0)
    (def deliver 0)
    (def send (make-hash-table-eq))

    (for (evt (unbox traces))
      (match evt
        (['trace ts src dest [what . _]]
         (hash-update! send what 1+ 0))
        (['publish . _]
         (set! publish (1+ publish)))
        (['deliver . _]
         (set! deliver (1+ deliver)))))

    (displayln "=== simulation summary ===")
    (displayln "nodes: " nodes)
    (displayln "messages: " messages)
    (displayln "fanout: " fanout)
    (displayln "publish: " publish)
    (displayln "deliver: " deliver)
    (for ((values msg count) send)
      (displayln msg ": " count)))

  (let (simulator (start-simulation! script: my-script
                                     trace: my-trace
                                     router: router
                                     params: params
                                     nodes: nodes
                                     N-connect: connect))
    (thread-join! simulator)
    (display-summary!)
    (transcript (unbox traces))))

(def (save-transcript-to-file file)
  (lambda (trace)
    (let (trace (reverse trace))
      (call-with-output-file file
        (lambda (port)
          (parameterize ((current-output-port port))
            (for-each displayln trace)))))))
