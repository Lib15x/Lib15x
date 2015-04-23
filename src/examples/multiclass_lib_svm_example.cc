#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>
#include <kernels/KernelDot.hpp>
#include <models/MulticlassClassifier.hpp>
#include <preprocessing/MinMaxScaler.hpp>

using namespace CPPLearn;

using Scaler=Preprocessing::MinMaxScaler;
using Kernel=Kernels::RBF;
using BinaryModel=Models::LibSVM<Kernel>;
using MulticlassModel=Models::MulticlassClassifier<BinaryModel>;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  string trainfilename="../../data/libsvm/glass.scale.cl";
  string testfilename="../../data/libsvm/glass.scale.cl";

  std::pair<MatrixXd, VectorXd> trainPair=
    Utilities::readCPPLearnDataFile(trainfilename);

  std::pair<MatrixXd, VectorXd> testPair=
    Utilities::readCPPLearnDataFile(testfilename);

  size_t numberOfFeatures=trainPair.first.cols();
  Scaler scaler;
  MatrixXd trainData=scaler.fitTransform(trainPair.first);
  MatrixXd testData=scaler.transform(testPair.first);

  double gamma=1.0/numberOfFeatures;
  Kernel kernel{gamma};
  double C=1.0;
  BinaryModel binaryModel{kernel, numberOfFeatures,C};

  unsigned numberOfClasses=trainPair.second.maxCoeff()+1;
  MulticlassModel multiclassModel{numberOfClasses, binaryModel};

  clock_t t;
  t=clock();

  multiclassModel.whetherVerbose() = VerboseFlag::Verbose;

  multiclassModel.train(trainData, trainPair.second);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);

  t=clock();
  VectorXd predictedLabels=multiclassModel.predict(testData);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);

  unsigned numberOfMatches=0;
  for (unsigned index=0; index<testPair.second.size(); ++index){
    numberOfMatches+=(predictedLabels[index]==testPair.second[index]);
  }

  double accuracy=double(numberOfMatches)/testPair.second.size()*100;
  printf("accuracy = %f%%, (%u / %lu)\n", accuracy, numberOfMatches, testPair.second.size());

  return 0;
}
