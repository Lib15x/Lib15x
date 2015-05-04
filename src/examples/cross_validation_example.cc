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
  string trainfilename="../../data/libsvm/train.2.cl";

  auto trainPair= Utilities::readCPPLearnDataFile(trainfilename);
  MatrixXd& trainData=trainPair.first;
  Labels& trainLabels=trainPair.second;

  Scaler scaler;
  trainData=scaler.fitTransform(trainData);

  CrossValidation crossValidation{trainData, trainLabels, true};

  size_t numberOfFeatures=trainData.cols();
  double gamma=0.5;

  Kernel kernel{gamma};
  double C = 2.0;
  BinaryModel binaryModel{kernel, numberOfFeatures,C};
  size_t numberOfClasses=(size_t)trainLabels.labelData.maxCoeff()+1;
  MulticlassModel multiclassModel{numberOfClasses, binaryModel};

  VectorXd losses=crossValidation.computeValidationLosses(&multiclassModel);
  cout<<1-losses.mean()<<endl;

  return 0;
}
