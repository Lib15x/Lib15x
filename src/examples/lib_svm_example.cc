#include <Lib15x.hpp>
using namespace Lib15x;

using Kernel=Kernels::RBF;
using LearningModel=Models::LibSVM<Kernel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  const string trainfilename="../../data/test/libsvm_train_1.cl";

  const std::pair<MatrixXd, Labels> trainPair= Utilities::readLib15xDataFile(trainfilename);
  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;

  const long numberOfData=trainData.rows();
  const long numberOfFeatures=trainData.cols();
  const double gamma=1.0/static_cast<double>(numberOfFeatures);

  const long numberOfClasses = static_cast<long>(trainLabels._labelData.maxCoeff())+1;

  LearningModel learningModel{numberOfFeatures, numberOfClasses, Kernel{gamma}};
  learningModel.whetherVerbose()=VerboseFlag::Verbose;

  clock_t t;
  t=clock();
  learningModel.train(trainData, trainLabels);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",
          t, (double)t/CLOCKS_PER_SEC);

  cout<<"number of support vectors: "<<learningModel.getSupportVectors().rows()<<endl;

  t=clock();
  Labels predictedLabels=learningModel.predict(trainPair.first);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",
          t, (double)t/CLOCKS_PER_SEC);

  double accuracy=1.0-LossFunction(predictedLabels, trainPair.second)
    /static_cast<double>(numberOfData);
  printf("accuracy = %f%%, (%ld / %ld)\n", accuracy*100,
         (long)(accuracy*(double)numberOfData), numberOfData);

  return 0;
}
