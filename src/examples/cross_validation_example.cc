#include <core/Definitions.hpp>
#include <models/LinearRidgeRegression.hpp>
#include <validation/CrossValidation.hpp>

using namespace CPPLearn;
using LearningModel=Models::LinearRidgeRegression;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,0.1);
  size_t numberOfData = 200;
  size_t numberOfFeatures = 10;
  MatrixXd data=MatrixXd::Random(numberOfData, numberOfFeatures);
  VectorXd parameters=VectorXd::Random(numberOfFeatures);





  VectorXd labels=data*parameters;
  for (size_t index=0; index<numberOfData; ++index)
    labels(index) += distribution(generator);

  double regularizer=0;
  LearningModel linearRidgeRegression(regularizer);
  CrossValidation<LearningModel> crossValidation{linearRidgeRegression};
  try{
    VectorXd scores= crossValidation.computeValidationScores(data,labels);
    cout<<scores<<endl;
  } catch (std::exception& e){
    cout<<"there is a problem when doing cross validation!"<<endl;
    throw;
  }
}
