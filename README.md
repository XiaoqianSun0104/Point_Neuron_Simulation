# Simulation Package
Spiking-level simulation is a computational approach to modeling neural activity at the level of individual action potentials. Unlike rate-based models, which approximate neural activity as a continuous firing rate, spiking models explicitly represent discrete spike events over time.
One of the most commonly used spiking neuron models is the Leaky Integrate-and-Fire (LIF) neuron, which captures essential neuronal dynamics with computational efficiency. This package implements the spiking-level LIF point neuron simulations.


## Overview
### Neuron
- Creates a neuron with all biophysical properties (e.g., exc/inh/leak conductance level, time constant, membrane potential related - V_{rest}, V_{init}, V_{leakReversal}, V_{excReversal}, V_{fireThreshold}...)
- Supports pre-assigned spike trains attached to a neuron
- Supports pre-generated external inputs with various shapes

### Connectivity
- Defines synaptic connections between neurons
- Implements different connectivity rules (e.g., random, structured, weight tuning)

### Simulation Object
- Runs the simulation, evolving neural activity over time
- Handles synaptic interactions and spike propagation
- To turn off synaptic plasticity, simply set learning rates to 0 

### Functional Modules

#### signal.py
- Generates different types of input signals 
    - Poisson Spike Train
    - Sinusoidal
    - Noisy following Ornstein-Uhlenbeck process
- Modify SpkTrain by absolute refractory period
- Add bursting activity

#### utils.py
- Firing evaluation (ISI, CV, Bursting Level...)
- CCG, GLMCC, GLMPP 
- Extract info from objects

#### visualization.py
- Provides functions for plotting simulation results

### Use Case Folder
- Jupyternotebook examples


