#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>

using namespace CPPLearn;

using Kernel=Kernels::RBF;
using LearningModel=Models::LibSVM<Kernel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  string trainfilename="../../data/test/libsvm_train_1.cl";

  auto trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;

  size_t numberOfData=trainData.rows();
  size_t numberOfFeatures=trainData.cols();
  double gamma=1.0/numberOfFeatures;

  Kernel kernel{gamma};
  LearningModel learningModel{kernel, numberOfFeatures};

  clock_t t;
  t=clock();
  learningModel.train(trainData, trainLabels);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);
  cout<<"number of support vectors: "<<learningModel.getSupportVectors().rows()<<endl;

  t=clock();
  Labels predictedLabels=learningModel.predict(trainPair.first);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);

  double accuracy=1.0-LossFunction(predictedLabels, trainPair.second)/numberOfData;
  printf("accuracy = %f%%, (%u / %lu)\n", accuracy*100,
         (unsigned)(accuracy*numberOfData), numberOfData);

  return 0;
}
