#include <Lib15x.hpp>
using namespace Lib15x;

using LearningModel=Models::LinearRidgeRegression;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,0.1);
  long numberOfData = 20;
  long numberOfFeatures = 10;
  MatrixXd data = MatrixXd::Random(numberOfData, numberOfFeatures);
  VectorXd parameters = VectorXd::Random(numberOfFeatures);

  Labels labels{ProblemType::Regression};
  labels._labelData.resize(numberOfData);
  labels._labelData=data*parameters;
  for (long index=0; index<numberOfData; ++index)
    labels._labelData(index) += distribution(generator);

  double regularizer = 0.0;
  LearningModel learningModel{numberOfFeatures, regularizer};

  learningModel.train(data, labels);

  Labels predictedLabels=learningModel.predict(data);
  double loss=LossFunction(predictedLabels, labels);
  cout<<loss<<endl;
}
