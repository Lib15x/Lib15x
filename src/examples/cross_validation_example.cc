#include <Lib15x.hpp>
using namespace Lib15x;

using Scaler=Preprocessing::MinMaxScaler;
using Kernel=Kernels::RBF;
using BinaryModel=Models::LibSVM<Kernel>;
using MulticlassModel=Models::MulticlassClassifier<BinaryModel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  string trainfilename = "../../data/libsvm/train.2.cl";

  std::pair<MatrixXd, Labels> trainPair = Utilities::readLib15xDataFile(trainfilename);
  MatrixXd& trainData = trainPair.first;
  Labels& trainLabels = trainPair.second;

  Scaler scaler;
  trainData = scaler.fitTransform(trainData);

  const CrossValidation crossValidation{trainData, trainLabels, true};

  const long numberOfFeatures=trainData.cols();
  double gamma = 0.5;

  Kernel kernel{gamma};
  double C = 2.0;
  BinaryModel binaryModel{numberOfFeatures,2, C, kernel};
  long numberOfClasses = static_cast<long>(trainLabels._labelData.maxCoeff())+1;
  MulticlassModel multiclassModel{numberOfFeatures, numberOfClasses, binaryModel};

  VectorXd losses=crossValidation.computeValidationLosses(&multiclassModel);
  cout<<"cross validation error ="<<losses.mean()<<endl;

  return 0;
}
