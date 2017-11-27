# Digit Recognizer
## Intrduction
Digit Recognizer is a C++ project that uses the Intel® Data Analytics Acceleration Library (Intel® DAAL)<br/>
to recognize handwritten digits. This is a tipical type of machine learning problem that can be solved in many different ways.<br/>
This solution to the problem uses Support Vector Machine (SVM) to demonstrate how the library algorithm in Intel® DAAL can be used to solve this problem.
## Project Description
Given a handwritten digit, the system should be able to recognize or infer what digit was written.<br/>
For the system to be able to predict the output with a given input, it needs a trained model learned from the training data set that provides the system with the capability to make an inference or prediction.
## The Plan
Since me and my partner have no previous experience in Machine Learning we have decided to break down the project into smaller tasks described below.
### Loading Data in Intel DAAL
To load some data to be used to train our model we will use public data downloadable from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits?utm_campaign=cmd_12617-1&utm_source=pum26&utm_medium=pdf&utm_content=zhu_uci_machinelearning_link1 "UCI Repository"). This data set contains data to train and test the model.<br/>
The data set was donated on 1998-07-01. 43 People contributed to it. 30 People contributed to the training set and 13 different people contributed to the test set.
For this project the data will be loaded using the CSV format. The training data is stored in a file named *digits_tra.csv* and the test data will be stored in a file name *digits_tes.csv*.<br/>
## Group Members
1. Rosario A. Cali
2. [Joseph Pildush](https://github.com/jpil101 "Joseph's GitHub Homepage")
