# The learning algorithm of Control Barrier functions with application in bipedal robots

This repo is only for testing the naive idea of the algorithm in a toy environment. 
To see the experiment in robot environment, please go to [yangcyself/BYSJ02](https://github.com/yangcyself/bysj02)

## Dependices

- scikit-learn
- scipy
- autograd
- ExperimentSecretary
  - An self-developed utility for logging and analyzing experiment settings and results
  - [github](https://github.com/yangcyself/ExperimentSecretary.git)

## Repo structure

The pipeline of the algorithm contains 3 steps:
1. sample
   1. use [`sample`](./sample.py) to generate the positive and negative samples about the safety set.
2. classification
   1. use an SVM like algorithm to classify the samples from the previous step, to get a quadratic form CBF
3. CBF_CLF_QP
   1. Put the CBF of the previous step into CBF_CLF_QP and run simulation