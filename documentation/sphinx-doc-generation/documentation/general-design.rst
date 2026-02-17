General Design of Sentra
====================================

.. plantuml::

    skinparam BackgroundColor #FFFFFF00

    participant SentraBackend
    participant SentraNode1
    participant SentraNode2

    SentraNode1 --> SentraBackend: TLS Channel Establishment
    SentraNode2 --> SentraBackend: TLS Channel Establishment
    SentraBackend --> SentraNode1: Attestion request
    SentraBackend --> SentraNode2: Attestion request
    SentraNode1 --> SentraBackend: Attestion Report
    SentraNode2 --> SentraBackend: Attestion Report

