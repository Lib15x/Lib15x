#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/LibSVM.hpp>
#include <gtest/gtest.h>

using namespace CPPLearn;

TEST(LibSVM, RBFKernel_test)
{
  using Kernel=Kernels::RBF;
  using LearningModel=Models::LibSVM<Kernel>;

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
  long numberOfFeatures=data.cols();
  long numberOfData=data.rows();

  Labels labels{ProblemType::Classification};
  labels._labelData.resize(numberOfData);
  labels._labelData <<1, 1, 1, 1, 1, 0, 0, 0, 0;
  long numberOfClasses = 2;

  double gamma=1.0/static_cast<double>(numberOfFeatures);
  Kernel kernel{gamma};
  LearningModel learningModel{numberOfFeatures, numberOfClasses, 1e5, kernel};

  learningModel.train(data, labels);
  Labels predictedLabels=learningModel.predict(data);

  for (long testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(labels._labelData[testIndex], predictedLabels._labelData(testIndex));
}

TEST(LibSVM, DotKernel_test)
{
  using Kernel=Kernels::Dot;
  using LearningModel=Models::LibSVM<Kernel>;

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

  long numberOfFeatures=data.cols();
  long numberOfData=data.rows();

  Labels labels{ProblemType::Classification};
  labels._labelData.resize(numberOfData);
  labels._labelData<<1, 1, 1, 1, 1, 0, 0, 0, 0;
  long numberOfClasses=2;

  Kernel kernel{ };
  LearningModel learningModel{numberOfFeatures, numberOfClasses, 1e5, Kernel()};

  learningModel.train(data, labels);
  Labels predictedLabels=learningModel.predict(data);

  for (long testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(labels._labelData[testIndex], predictedLabels._labelData(testIndex));
}

TEST(LibSVM, copy_test)
{
  //using Kernel=Kernels::RBF;
  //using LearningModel=Models::LibSVM<Kernel>;
  //
  //MatrixXd data(10,5) = random;
  //data<< 1,1,
  //1, 3,
  //2, 2,
  //2, 4,
  //3, 2,
  //3, 4,
  //4, 1,
  //4, 3,
  //5, 5;
  //
  //long numberOfFeatures=data.cols();
  //long numberOfData=data.rows();
  //
  //Labels labels{ProblemType::Classification};
  //labels._labelData.resize(numberOfData);
  //labels._labelData<<1, 1, 1, 1, 1, 0, 0, 0, 0;
  //
  //Kernel kernel{ };
  //LearningModel learningModel{kernel, numberOfFeatures, 1e5};
  //
  //learningModel.train(data, labels);
  //Labels predictedLabels=learningModel.predict(data);
  //
  //for (long testIndex=0; testIndex<numberOfData; ++testIndex)
  //EXPECT_EQ(labels._labelData[testIndex], predictedLabels._labelData(testIndex));
}
