#include <core/Definitions.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <validation/CrossValidation.hpp>
#include <models/MulticlassClassifier.hpp>
#include <preprocessing/MinMaxScaler.hpp>

using namespace CPPLearn;

using Scaler=Preprocessing::MinMaxScaler;
using Kernel=Kernels::RBF;
using BinaryModel=Models::LibSVM<Kernel>;
using MulticlassModel=Models::MulticlassClassifier<BinaryModel>;

int main(int argc, char* argv[])
{
  ignoreUnusedVariables(argc, argv);
  string trainfilename = "../../data/libsvm/train.2.cl";

  std::pair<MatrixXd, Labels> trainPair = Utilities::readCPPLearnDataFile(trainfilename);
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
  cout<<1-losses.mean()<<endl;

  return 0;
}
