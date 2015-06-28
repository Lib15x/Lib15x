#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/TreeRegressor.hpp>

using namespace CPPLearn;
using LearningModel=Models::TreeRegressor;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  MatrixXd trainData=MatrixXd::Random(5,1);
  Labels trainLabels{ProblemType::Regression};
  trainLabels._labelData = VectorXd::Random(5);

  const long numberOfFeatures=trainData.cols();

  LearningModel learningModel{numberOfFeatures};
  learningModel.whetherVerbose()=VerboseFlag::Verbose;

  learningModel.train(trainData, trainLabels);
  Labels predictedLabels=learningModel.predict(trainData);

  double loss=LossFunction(predictedLabels, trainLabels);
  cout<<"classification error on training set = "<<loss<<endl;
  learningModel.clear();

  return 0;
}
