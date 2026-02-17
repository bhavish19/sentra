General Design of Sentra
====================================

.. plantuml::

    skinparam BackgroundColor #FFFFFF00

    participant SentraBackend
    participant SentraNode1
    participant SentraNode2

    group Candidate regstration
    SentraNode1 --> SentraBackend: TLS Channel Establishment
    SentraNode2 --> SentraBackend: TLS Channel Establishment
    SentraBackend --> SentraNode1: Attestion request
    SentraBackend --> SentraNode2: Attestion request
    SentraNode1 --> SentraBackend: Attestion Report
    SentraNode2 --> SentraBackend: Attestion Report

    group Commitee selection
    activate SentraBackend
    SentraBackend --> SentraBackend: Check the nodes to make them candidates (trust score, attested?)
    SentraBackend --> SentraBackend: Check if there are enough candidates
    SentraBackend --> SentraBackend: Can we reuse a prvious Commitee?
    SentraBackend --> SentraBackend: Sort nodes according to trust score
    SentraBackend --> SentraBackend: Execute the greedy algorithm to filter according to CPU, Host, Operator




