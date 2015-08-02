# The Lib15x machine learning library

Lib15x is a machine learning library that implements supervised learning algorithms covered in Caltech machine learning class cs155 and cs156. The library is written in C++ and uses curiously recurring template pattern (CRTP) as the basic design structure in order to take advantage of compile time polymorphism for speed but at the same time still able to explore class hierarchies to ease implementation.

The purpose of the library is not to compete with any existing machine learning libraries, but to serve as a channel to experiment different design ideas and concurrency techniques to deal with large data set. Please refer to "The designing idea behind Lib15x" for detailed explanation for the designing concerns of the library.

##Lib15x currently implements the following models:
* **data preprocessing:** minmax scaler, standard scaler.
* **basic models:** ridge regression, logistic regression, support vector machine, decision tree, random forest.
* **higher level methods:** bootstrap aggregation, gradient boosting, multiclass classification, cross validation.