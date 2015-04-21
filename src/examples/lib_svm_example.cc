#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/LibSVM.hpp>
#include <kernels/KernelRBF.hpp>

using namespace CPPLearn;

using Kernel=Kernels::RBF;
using LearningModel=Models::LibSVM<Kernel>;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);

  string trainfilename="../../data/test/libsvm_train_1.cl";

  std::pair<MatrixXd, VectorXd> trainPair=
    Utilities::readCPPLearnDataFile(trainfilename);

  size_t numberOfFeatures=trainPair.first.cols();
  double gamma=1.0/numberOfFeatures;

  Kernel kernel{gamma};
  LearningModel learningModel{kernel, numberOfFeatures};

  clock_t t;
  t=clock();
  learningModel.train(trainPair.first, trainPair.second);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for training.\n",t,((float)t)/CLOCKS_PER_SEC);
  cout<<"number of support vectors: "<<learningModel.getSupportVectors().rows()<<endl;

  t=clock();
  VectorXd predictedLabels=learningModel.predict(trainPair.first);
  t=clock()-t;
  printf ("It took me %ld clicks (%f seconds) for predicting.\n",t,((float)t)/CLOCKS_PER_SEC);
  cout<<predictedLabels<<endl;

  double mismatch=(predictedLabels-trainPair.second).squaredNorm();
  cout<<mismatch/trainPair.second.size()<<endl;
  return 0;
}
