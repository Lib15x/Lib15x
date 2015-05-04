#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LinearLogisticRegression.hpp>

using namespace CPPLearn;

using LearningModel=Models::LinearLogisticRegression;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  string trainfilename="../../data/libsvm/heart_scale.cl";

  auto trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;

  size_t numberOfData=trainData.rows();
  size_t numberOfFeatures=trainData.cols();

  const Penalty penaltyType = Penalty::L1;
  LearningModel learningModel{numberOfFeatures, penaltyType};

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
  printf("accuracy = %f%%, (%u / %lu)\n", accuracy*100,
         (unsigned)(accuracy*(double)numberOfData), numberOfData);

  return 0;
}
