#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/MulticlassClassifier.hpp>

using namespace CPPLearn;

using Kernel=Kernels::RBF;
using BinaryModel=Models::LibSVM<Kernel>;
using MulticlassModel=Models::MulticlassClassifier<BinaryModel>;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  string trainfilename="../../data/libsvm/train.2.cl";

  std::pair<MatrixXd, VectorXd> trainPair=
    Utilities::readCPPLearnDataFile(trainfilename);

  size_t numberOfFeatures=trainPair.first.cols();
  double gamma=1.0/numberOfFeatures;

  Kernel kernel{gamma};
  BinaryModel binaryModel{kernel, numberOfFeatures};
  MulticlassModel multiclassModel{3, binaryModel};

  clock_t t;
  t=clock();
  for (unsigned index=0; index<trainPair.second.size(); ++index)
    trainPair.second[index]-=1;

  multiclassModel.whetherVerbose() = VerboseFlag::Verbose;

  multiclassModel.train(trainPair.first, trainPair.second);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);

  t=clock();
  VectorXd predictedLabels=multiclassModel.predict(trainPair.first);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);

  unsigned numberOfMatches=0;
  for (unsigned index=0; index<trainPair.second.size(); ++index){
    numberOfMatches+=(predictedLabels[index]==trainPair.second[index]);
  }

  double accuracy=double(numberOfMatches)/trainPair.second.size()*100;
  printf("accuracy = %f%%, (%u / %lu)\n", accuracy, numberOfMatches, trainPair.second.size());

  return 0;
}
