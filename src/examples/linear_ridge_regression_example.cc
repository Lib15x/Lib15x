#include <core/Definitions.hpp>
#include <models/LinearRidgeRegression.hpp>

using namespace CPPLearn;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,0.1);
  size_t numberOfData = 20;
  size_t numberOfFeatures = 10;
  MatrixXd data=MatrixXd::Random(numberOfData, numberOfFeatures);
  VectorXd parameters=VectorXd::Random(numberOfFeatures);
  VectorXd labels=data*parameters;
  for (size_t index=0; index<numberOfData; ++index)
    labels(index) += distribution(generator);

  double regularizer=0;
  LinearRidgeRegression linearRidgeRegression(regularizer);

  try{
    linearRidgeRegression.train(data, labels);
  } catch (std::exception &e){
    cout<<"there is a problem when training linear regression model!"<<endl;
    throw;
  }

  VectorXd predictedLabel=linearRidgeRegression.predict(data);
  cout<<(predictedLabel-labels).norm()<<endl;
}
