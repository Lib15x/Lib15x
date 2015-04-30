#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/MulticlassClassifier.hpp>
#include <preprocessing/MinMaxScaler.hpp>

using namespace CPPLearn;

using Scaler=Preprocessing::MinMaxScaler;
using Kernel=Kernels::RBF;
using BinaryModel=Models::LibSVM<Kernel>;
using MulticlassModel=Models::MulticlassClassifier<BinaryModel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=MulticlassModel::LossFunction;

  string trainfilename="../../data/libsvm/glass.scale.cl";
  string testfilename="../../data/libsvm/glass.scale.cl";

  auto trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  auto testPair= Utilities::readCPPLearnDataFile(testfilename);

  MatrixXd& trainData=trainPair.first;
  Labels& trainLabels=trainPair.second;
  MatrixXd& testData=testPair.first;
  Labels& testLabels=testPair.second;

  size_t numberOfTestData=testData.rows();

  size_t numberOfFeatures=trainData.cols();
  Scaler scaler;
  trainData=scaler.fitTransform(trainPair.first);
  testData=scaler.transform(testPair.first);

  double gamma=1.0/(double)numberOfFeatures;
  Kernel kernel{gamma};
  double C=1.0;
  BinaryModel binaryModel{kernel, numberOfFeatures,C};

  size_t numberOfClasses=(size_t)trainLabels.labelData.maxCoeff()+1;
  MulticlassModel multiclassModel{numberOfClasses, binaryModel};

  clock_t t;
  t=clock();

  multiclassModel.whetherVerbose() = VerboseFlag::Verbose;

  multiclassModel.train(trainData, trainLabels);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);

  t=clock();
  Labels predictedLabels=multiclassModel.predict(testData);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);


  double loss=LossFunction(predictedLabels, testLabels);
  double accuracy=1-loss/(double)numberOfTestData;
  printf("accuracy = %f%%, (%lu / %lu)\n", accuracy*100,
         (size_t)(accuracy*(double)numberOfTestData), numberOfTestData);

  return 0;
}
