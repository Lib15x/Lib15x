#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LinearLogisticRegression.hpp>

using namespace Lib15x;

using LearningModel=Models::LinearLogisticRegression;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  string trainfilename="../../data/libsvm/heart_scale.cl";

  std::pair<MatrixXd, Labels> trainPair= Utilities::readLib15xDataFile(trainfilename);
  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;

  const long numberOfData=trainData.rows();
  const long numberOfFeatures=trainData.cols();
  const long numberOfClasses = static_cast<long>(trainLabels._labelData.maxCoeff())+1;

  const Penalty penaltyType = Penalty::L1;
  LearningModel learningModel{numberOfFeatures, numberOfClasses, penaltyType};
  learningModel.whetherVerbose()=VerboseFlag::Verbose;

  clock_t t;
  t=clock();
  learningModel.train(trainData, trainLabels);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);

  t=clock();
  Labels predictedLabels=learningModel.predict(trainPair.first);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);

  double accuracy=1.0-LossFunction(predictedLabels, trainPair.second)/(double)numberOfData;
  printf("accuracy = %f%%, (%ld / %ld)\n", accuracy*100,
         (long)(accuracy*(double)numberOfData), numberOfData);


  return 0;
}
