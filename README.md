# CCNETS 

CCNETS is uniquely crafted to emulate brain-like information processing and comprises three main components: explainer, producer, and reasoner. Each component is designed to mimic specific brain functions, which aids in generating high-quality datasets and enhancing the classification performance


## Installation
To use CCNETS , you can clone this repository:
```bash
git clone https://github.com/hanbeotPark/CCNETS.git
```

## Usage

Here is a basic example of how to use CCNETS and SupervisedLearningwithCCNETs:

```python
from ccnets.ccnets import CCNets
from ccnets.supervised_learning_with_ccnets import SupervisedLearningWithCCNets

# Create a dataset
trainset = Dataset(X_train, y_train)
testset = Dataset(X_test, y_test)

# Define CCNETS
ccnets = CCNets(args, model, model, model)

# Train CCNETS 
ccnets.train(trainset, testset)  

# Define SupervisedLearningwithCCNETs
sl_with_ccnets = SupervisedLearningWithCCNets(args, model)

# Train SupervisedLearningwithCCNETs
x,y = sl_with_ccnets.train(trainset, testset, ccnets, data_type)

# Perform prediction and validation
# ...

```
## Using Data
You can download imbalnced credit fraud dataset here:
* https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

## COPYRIGHT
COPYRIGHT (c) 2022. CCNets. All Rights reserved
