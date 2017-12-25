# Digit Recognizer
## Intrduction
Digit Recognizer is a C++ project that uses the Intel® Data Analytics Acceleration Library (Intel® DAAL)<br/>
to recognize handwritten digits. This is a tipical type of machine learning problems that can be solved in many different ways.<br/>
This solution to the problem uses Support Vector Machine (SVM), a supervised learning model, to demonstrate how the library algorithms in Intel® DAAL can be used to solve this problem.
## Project Description
Given a handwritten digit, the system should be able to recognize or infer what digit was written.<br/>
For the system to be able to predict the output with a given input, it needs a trained model learned from the training data set that provides the system with the capability to make an inference or prediction.
## The Plan
Since I am completely new to Machine Learning I have decided to break down the project into smaller tasks described below.
### Loading Data in Intel DAAL
To load some data to be used to train our model I will use public data downloadable from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits?utm_campaign=cmd_12617-1&utm_source=pum26&utm_medium=pdf&utm_content=zhu_uci_machinelearning_link1 "UCI Repository"). This data set contains data to train and test the model.<br/>
The data set was donated on 1998-07-01. 43 People contributed to it. 30 People contributed to the training set and 13 different people contributed to the test set.
For this project the data will be loaded using the CSV format. The training data is stored in a file named *digits_tra.csv* and the test data will be stored in a file name *digits_tes.csv*.<br/>
In order to load the data from the csv files we will be using some classes provided by the Intel®.<br/>
## External Resources
1. [Wiki Page](https://wiki.cdot.senecacollege.ca/wiki/Alpha_Centauri "Wiki Page")
2. [Code Example on Code Project](https://www.codeproject.com/Articles/1151612/A-Performance-Library-for-Data-Analytics-and-Machi "Code Project")
3. [Related question on the Intel Forum's page](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library/topic/749376#comment-1915946 "Intel Forum")
