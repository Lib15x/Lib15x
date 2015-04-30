#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/SupportVectorClassifier.hpp>
#include <gtest/gtest.h>

using namespace CPPLearn;

TEST(SupportVectorClassifier, RBFKernel_test)
{
  using Kernel=Kernels::RBF;
  using LearningModel=Models::SupportVectorClassifier<Kernel>;

  MatrixXd data(9,2);
  data<< 1,1,
    1, 3,
    2, 2,
    2, 4,
    3, 2,
    3, 4,
    4, 1,
    4, 3,
    5, 5;

  Labels labels{ProblemType::Classification};
  labels.labelData.resize(9);
  labels.labelData<<1, 1, 1, 1, 1, 0, 0, 0, 0;

  size_t numberOfFeatures=data.cols();
  size_t numberOfData=data.rows();

  double gamma=1.0/numberOfFeatures;
  Kernel kernel{gamma};
  LearningModel learningModel{kernel, numberOfFeatures, 1e5};

  learningModel.train(data, labels);
  Labels predictedLabels=learningModel.predict(data);

  for (size_t testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(labels.labelData[testIndex], predictedLabels.labelData(testIndex));
}

TEST(SupportVectorClassifier, DotKernel_test)
{
  using Kernel=Kernels::Dot;
  using LearningModel=Models::SupportVectorClassifier<Kernel>;

  MatrixXd data(9,2);
  data<< 1,1,
    1, 3,
    2, 2,
    2, 4,
    3, 2,
    3, 4,
    4, 1,
    4, 3,
    5, 5;

  Labels labels{ProblemType::Classification};
  labels.labelData.resize(9);
  labels.labelData<<1, 1, 1, 1, 1, 0, 0, 0, 0;

  size_t numberOfFeatures=data.cols();
  size_t numberOfData=data.rows();

  double C=1e5;
  Kernel kernel;
  LearningModel learningModel{kernel, numberOfFeatures, C};

  learningModel.train(data, labels);
  Labels predictedLabels=learningModel.predict(data);

  for (size_t testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(labels.labelData[testIndex], predictedLabels.labelData(testIndex));
}
