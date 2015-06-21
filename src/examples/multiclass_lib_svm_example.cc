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
  const string trainfilename = "../../data/libsvm/glass.scale.cl";
  const string testfilename = "../../data/libsvm/glass.scale.cl";

  std::pair<MatrixXd, Labels> trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  std::pair<MatrixXd, Labels> testPair= Utilities::readCPPLearnDataFile(testfilename);

  MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;
  MatrixXd& testData=testPair.first;
  const Labels& testLabels=testPair.second;

  const long numberOfTestData=testData.rows();

  const long numberOfFeatures=trainData.cols();
  Scaler scaler;
  trainData=scaler.fitTransform(trainPair.first);
  testData=scaler.transform(testPair.first);

  const double gamma=1.0/static_cast<double>(numberOfFeatures);
  const Kernel kernel{gamma};
  const double C=1.0;
  BinaryModel binaryModel{numberOfFeatures, 2, C, kernel};

  const long numberOfClasses = static_cast<long>(trainLabels._labelData.maxCoeff())+1;
  //MulticlassModel multiclassModel{numberOfFeatures, numberOfClasses, binaryModel};
  MulticlassModel multiclassModel{numberOfFeatures, numberOfClasses, binaryModel};

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
  printf("accuracy = %f%%, (%ld / %ld)\n", accuracy*100,
         (long)(accuracy*(double)numberOfTestData), numberOfTestData);

  return 0;
}
