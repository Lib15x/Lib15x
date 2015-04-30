#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/SupportVectorClassifier.hpp>

using namespace CPPLearn;

using Kernel=Kernels::RBF;
using LearningModel=Models::SupportVectorClassifier<Kernel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  string trainfilename="../../data/test/libsvm_train_1.cl";
  string testfilename="../../data/test/libsvm_test_1.cl";
  string labelfilename="../../data/test/libsvm_1.output";

  auto trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  auto testPair= Utilities::readCPPLearnDataFile(testfilename);

  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;
  const MatrixXd& testData=testPair.first;
  const Labels& testLabels=testPair.second;

  const size_t numberOfTestData=testData.rows();

  size_t numberOfFeatures=trainData.cols();
  double gamma=1.0/(double)numberOfFeatures;

  Kernel kernel{gamma};
  LearningModel learningModel{kernel, numberOfFeatures};
  learningModel.printOptimizationProgress()=true;
  clock_t t;
  t=clock();
  learningModel.train(trainData, trainLabels);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);
  cout<<"number of support vectors: "<<learningModel.getSupportVectors().rows()<<endl;

  t=clock();
  Labels predictedLabels=learningModel.predict(testData);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);

  double accuracy=1.0-LossFunction(predictedLabels, testLabels)/(double)numberOfTestData;
  printf("accuracy = %f%%, (%u / %lu)\n", accuracy*100,
         (unsigned)(accuracy*(double)numberOfTestData), numberOfTestData);
  return 0;
}
