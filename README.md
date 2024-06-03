# ProtoMix
This is a Pytorch implementation of ProtoMix: Augmenting Health Status Representation Learning via Prototype-based Mixup.
More details of the paper and dataset will be released after it is published.


# The Code

## Requirements

Following the suggested way by PyHealth: https://github.com/sunlabuiuc/PyHealth. Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── models                  # The core source code of our model ProtoMix
    │   |──  RNN.py             # The Backbone of RNN
    │   |──  ProtoMix.py        # The main model of ProtoMix 
    │   |──  model_utils.py     # Utils file
    │   |──  run_utils.py       # Utils file
    ├── parser.py               # Parser Args
    ├── mian_protomix.py        # This is the main file
    └── README.md               # This document
```

## Datasets

Download MIMIC-III & MIMIC-IV & eICU datasets from https://physionet.org/content/; 

