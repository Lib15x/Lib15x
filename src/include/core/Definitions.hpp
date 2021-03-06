// -*- C++ -*-
#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#include <cmath>
#include <cstdio>
#include <vector>
#include <map>
#include <set>
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


namespace Lib15x
{
  using std::array;
  using std::vector;
  using std::string;
  using std::ifstream;
  using std::cout;
  using std::endl;

  using MatrixXd=Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Eigen::VectorXd;
  using Eigen::Map;

  enum class VerboseFlag {Quiet, Verbose};
  enum class ProblemType {Classification, Regression};
  enum class Penalty {L1, L2};

#pragma GCC diagnostic ignored "-Wunused-parameter"
  template <typename Variable>
  void ignoreUnusedVariable(Variable dummy) {}
  template <typename T>
  void ignoreUnusedVariables(const T & t) {}
  template <typename T, typename U>
  void ignoreUnusedVariables(const T & t, const U & u) {}
  template <typename T, typename U, typename V>
  void ignoreUnusedVariables(const T & t, const U & u, const V & v) {}
  template <typename T, typename U, typename V, typename W>
  void ignoreUnusedVariables(const T & t, const U & u, const V & v, const W & w) {}

  char exceptionBuffer[10000];
#define throwException(s, ...)                  \
  sprintf(exceptionBuffer, s, ##__VA_ARGS__);   \
  throw std::runtime_error(exceptionBuffer);

  struct Labels
  {
    ProblemType _labelType;
    VectorXd _labelData;

    Labels(const ProblemType labelType) : _labelType{labelType}
    {
      if (_labelType!=ProblemType::Classification && labelType!=ProblemType::Regression){
        throwException("Error happened in label constructor: "
                       "Type of label mush be either Classification or Regression");
      }
    }

    long size() const {
      return _labelData.size();
    }
  };
}

#endif // DEFINITIONS_H
