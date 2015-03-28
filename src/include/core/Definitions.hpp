// -*- C++ -*-
#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>


enum VerboseFlag {Quiet, Verbose};

using std::array;
using std::vector;
using std::string;
using std::ifstream;
using std::cout;
using std::endl;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;


#pragma GCC diagnostic ignored "-Wunused-parameter"
template <typename Variable>
void ignoreUnusedVariable(Variable dummy) {
}
template <typename T>
void ignoreUnusedVariables(const T & t) {
}
template <typename T, typename U>
void ignoreUnusedVariables(const T & t, const U & u) {
}
template <typename T, typename U, typename V>
void ignoreUnusedVariables(const T & t, const U & u, const V & v) {
}
template <typename T, typename U, typename V, typename W>
void ignoreUnusedVariables(const T & t, const U & u, const V & v, const W & w) {
}
#endif // DEFINITIONS_H

