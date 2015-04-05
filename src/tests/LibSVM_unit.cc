#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <kernels/KernelRBF.hpp>
#include <models/LibSVM.hpp>
#include <gtest/gtest.h>

using namespace CPPLearn;

using Kernel=KernelRBF;
using LearningModel=LibSVM<Kernel>;

TEST(LibSVM, train_test) {
  string trainfilename="../../data/test/libsvm_train_1.cl";
  string testfilename="../../data/test/libsvm_test_1.cl";
  string labelfilename="../../data/test/libsvm_1.output";

  std::pair<MatrixXd, VectorXd> trainPair=
    Utilities::readCPPLearnDataFile(trainfilename);
  std::pair<MatrixXd, VectorXd> testPair=
    Utilities::readCPPLearnDataFile(testfilename);

  EXPECT_EQ(trainPair.first.cols(), testPair.first.cols());

  size_t numberOfFeatures=trainPair.first.cols();
  double gamma=1.0/numberOfFeatures;

  Kernel kernel{gamma};
  LearningModel learningModel{kernel, numberOfFeatures};

  learningModel.train(trainPair.first, trainPair.second);
  VectorXd predictedLabels=learningModel.predict(testPair.first);

  ifstream labelfile;
  labelfile.open(labelfilename);

  EXPECT_EQ(labelfile.is_open(),true);

  string line;
  vector<int> correctLabels;
  size_t numberOfTests=predictedLabels.size();
  while (labelfile.good()){
    getline(labelfile,line);
    if (line == "") break;
    correctLabels.push_back(atoi(line.c_str()));
  }

  EXPECT_EQ(correctLabels.size(),numberOfTests);

  for (size_t testIndex=0; testIndex<numberOfTests; ++testIndex)
    EXPECT_EQ(correctLabels[testIndex], predictedLabels(testIndex));
}
