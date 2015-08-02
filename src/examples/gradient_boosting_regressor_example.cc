#include <Lib15x.hpp>
using namespace Lib15x;

using LearningModel=Models::GradientBoostingRegressor;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;
  const string trainfilename="../../data/example/iris.cl";

  std::pair<MatrixXd, Labels> trainPair= Utilities::readLib15xDataFile(trainfilename);
  MatrixXd& trainData = trainPair.first;
  Labels& trainLabels = trainPair.second;
  trainLabels._labelType = ProblemType::Regression;

  const long numberOfFeatures=trainData.cols();
  const long numberOfTrees=10;

  LearningModel learningModel{numberOfFeatures, numberOfTrees};
  learningModel.whetherVerbose()=VerboseFlag::Verbose;
  learningModel.train(trainData, trainLabels);
  Labels predictedLabels=learningModel.predict(trainData);

  double loss=LossFunction(predictedLabels, trainLabels);
  cout<<"classification error on training set = "<<loss<<endl;

  return 0;
}
