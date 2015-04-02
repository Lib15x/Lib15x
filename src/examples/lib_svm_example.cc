#include <core/Definitions.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <kernels/KernelSigmoid.hpp>
#include <kernels/KernelPolynomial.hpp>

using namespace CPPLearn;

using Kernel=KernelRBF;
using LearningModel=LibSVM<Kernel>;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);
  double gamma = 0.1;
  //double r = 1;
  //unsigned d=2;
  Kernel kernel(gamma);
  double C=1.0;
  size_t numberOfFeatures=10;
  size_t numberOfData=20;
  size_t numberOfTests=5;

  LearningModel learningModel(kernel, C, numberOfFeatures);
  MatrixXd trainData=MatrixXd::Random(numberOfData, numberOfFeatures);
  VectorXd trainLabels=VectorXd::Random(numberOfData);

  MatrixXd testData=MatrixXd::Random(numberOfTests, numberOfFeatures);
  learningModel.train(trainData, trainLabels);
  VectorXd predictLables=learningModel.predict(testData);
  cout<<predictLables<<endl;

}
