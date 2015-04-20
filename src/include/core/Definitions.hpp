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
#include <time.h>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>


//make_unique function from effective morden c++ by Scott Meyers
namespace std{
  template<typename T, typename... Ts>
  unique_ptr<T> make_unique(Ts&&... params)
  {
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
  }
}

namespace CPPLearn{
  enum class VerboseFlag {Quiet, Verbose};
  using std::array;
  using std::vector;
  using std::string;
  using std::ifstream;
  using std::cout;
  using std::endl;
  using std::size_t;
  using std::shared_ptr;
  using std::unique_ptr;
  using std::make_shared;
  using std::make_unique;

  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::Map;

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

  char exceptionBuffer[10000];
#define throwException(s, ...)                  \
  sprintf(exceptionBuffer, s, ##__VA_ARGS__);   \
  throw std::runtime_error(exceptionBuffer);
}
#endif // DEFINITIONS_H
