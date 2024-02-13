# CCNETS 

CCNETS is uniquely crafted to emulate brain-like information processing and comprises three main components: explainer, producer, and reasoner. Each component is designed to mimic specific brain functions, which aids in generating high-quality datasets and enhancing the classification performance


## Installation
To use CCNETS , you can clone this repository:
```bash
git clone https://github.com/hanbeotPark/CCNETS.git
```

## Usage

The core functionality of Neucube Py revolves around the `reservoir` class, which represents the spiking neural network model. Here is a basic example of how to use Neucube Py:

```python
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.sampler import SpikeCount

# Create a Reservoir 
res = Reservoir(inputs=14)

# Convert data to spikes
X = Delta().encode_dataset(data)

# Simulate the Reservior
out = res.simulate(X)

# Extract state vectors from the spiking activity
state_vec = SpikeCount.sample(out)

# Perform prediction and validation
# ...

```

## COPYRIGHT
COPYRIGHT (c) 2022. CCNets. All Rights reserved
