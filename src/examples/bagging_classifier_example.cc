#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/TreeClassifier.hpp>
#include <models/BaggingClassifier.hpp>

using namespace CPPLearn;
using LearningModel=Models::BaggingClassifier<>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  constexpr double (*LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;
  const string trainfilename="../../data/example/iris.cl";

  const std::pair<MatrixXd, Labels> trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  const MatrixXd& trainData=trainPair.first;
  const Labels& trainLabels=trainPair.second;

  const long numberOfFeatures=trainData.cols();
  const long numberOfClasses = static_cast<long>(trainLabels._labelData.maxCoeff())+1;
  const long numberOfTrees=1;

  LearningModel learningModel{numberOfFeatures, numberOfClasses, numberOfTrees};
  learningModel.whetherVerbose()=VerboseFlag::Verbose;
  learningModel.train(trainData, trainLabels);
  Labels predictedLabels=learningModel.predict(trainData);

  double loss=LossFunction(predictedLabels, trainLabels);
  cout<<"classification error on training set = "<<loss<<endl;
  //cout<<predictedLabels._labelData<<endl;

  return 0;
}