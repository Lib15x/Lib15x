#include <core/Definitions.hpp>
#include <models/LinearRidgeRegression.hpp>
#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>

using namespace CPPLearn;
using LearningModel=Models::LinearRidgeRegression;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,0.1);
  size_t numberOfData = 20;
  size_t numberOfFeatures = 10;
  MatrixXd data=MatrixXd::Random(numberOfData, numberOfFeatures);
  VectorXd parameters=VectorXd::Random(numberOfFeatures);

  Labels labels{ProblemType::Regression};
  labels.labelData.resize(numberOfData);
  labels.labelData=data*parameters;
  for (size_t index=0; index<numberOfData; ++index)
    labels.labelData(index) += distribution(generator);

  double regularizer=0;
  LearningModel learningModel{regularizer,numberOfFeatures};

  learningModel.train(data, labels);

  Labels predictedLabels=learningModel.predict(data);
  double loss=LossFunction(predictedLabels, labels);
  cout<<loss<<endl;
}
