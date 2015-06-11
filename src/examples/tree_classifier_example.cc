#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/TreeClassifier.hpp>

using namespace CPPLearn;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  using LearningModel=Models::TreeClassifier;

  MatrixXd trainData(7,2);
  trainData<<14,0,
    10,1,
    13,0,
    8,1,
    11,0,
    9,1,
    10,0;

  const long numberOfFeatures=trainData.cols();
  const long numberOfData=trainData.rows();

  Labels labels{ProblemType::Classification};
  labels._labelData.resize(numberOfData);
  labels._labelData<<1,1,1,0,0,1,0;
  const long numberOfClasses=2;

  LearningModel learningModel{numberOfFeatures, numberOfClasses, 1, 1, 10};
  learningModel.train(trainData, labels);
  Labels predictedLabels=learningModel.predict(trainData);
  cout<<predictedLabels._labelData<<endl;

  return 0;
}
