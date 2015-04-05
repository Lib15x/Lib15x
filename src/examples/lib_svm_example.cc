#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <kernels/KernelSigmoid.hpp>
#include <kernels/KernelPolynomial.hpp>
#include <stdlib.h>

using namespace CPPLearn;

using Kernel=Kernels::RBF;
using LearningModel=Models::LibSVM<Kernel>;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  string inputfilename="../../data/libsvm/train.1";
  string outputfilename="../../data/libsvm/train.1.cl";
  Utilities::createCPPLearDataFileFromLibsvmFormat(inputfilename, outputfilename);

  std::pair<MatrixXd, VectorXd> trainPair=
    Utilities::readCPPLearnDataFile(outputfilename);

  size_t numberOfFeatures=trainPair.first.cols();

  double gamma=1.0/numberOfFeatures;
  Kernel kernel{gamma};

  LearningModel learningModel(kernel, numberOfFeatures);
  learningModel.train(trainPair.first, trainPair.second);
  VectorXd predictedLabels=learningModel.predict(trainPair.first);
  FILE* file=fopen("output.pre","w");
  if (!file) fprintf(file,"cannot open output file!");

  size_t numberOfTests=predictedLabels.rows();
  for (size_t i=0; i<numberOfTests; ++i)
    fprintf(file, "%d\n", (int)predictedLabels(i));
  fclose(file);

  return 0;
}
