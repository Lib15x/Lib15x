#include <core/Definitions.hpp>
#include <external/svm.h>
#include <external/svm.cpp>

namespace CPPLearn{
  template<class Kernel>
  class LibSVM{
  public:
    LibSVM(Kernel kernel_, double C_, size_t numberOfFeatures_, double tol_=1e-5) :
      kernel{kernel_}, numberOfFeatures{numberOfFeatures_}, svmModel{nullptr}, C(C_),
      numberOfTrainData{0}, tol{tol_} {}

    void train(const MatrixXd& trainData, const VectorXd& trainLabels) {
      if (trainData.cols() != numberOfFeatures)
        throw std::runtime_error("invalid inpute data, number of features mismatch!");

      if (trainData.rows() != trainLabels.size())
        throw std::runtime_error("data and label size mismatch!");

      numberOfTrainData = trainData.rows();

      libsvm::svm_parameter svmParameter;
      svmParameter.svm_type=libsvm::C_SVC;
      svmParameter.kernel_type=libsvm::PRECOMPUTED;
      svmParameter.C=C;
      svmParameter.eps=tol;
      svmParameter.cache_size=cacheSize;

      libsvm::svm_problem svmProblem;
      svmProblem.l=numberOfTrainData;
      svmProblem.y=new double[numberOfTrainData];
      svmProblem.x=new libsvm::svm_node*[numberOfTrainData];
      libsvm::svm_node* vector_x=new libsvm::svm_node[numberOfTrainData*(numberOfTrainData+1)];

      for (size_t dataIndex=0; dataIndex<numberOfTrainData; ++dataIndex){
        svmProblem.y[dataIndex]=trainLabels(dataIndex);
        svmProblem.x[dataIndex]=vector_x+(numberOfTrainData+1)*dataIndex;
        svmProblem.x[dataIndex]->index=0;
        svmProblem.x[dataIndex]->value=dataIndex+1;
      }

      for (size_t indexI=0; indexI<numberOfTrainData; ++indexI)
        for (size_t indexJ=0; indexJ<numberOfTrainData; ++indexJ){
          svmProblem.x[indexI][indexJ+1].index=indexJ+1;
          if (indexI <= indexJ)
            svmProblem.x[indexI][indexJ+1].value=kernel(trainData.row(indexI),
                                                        trainData.row(indexJ));
          else
            svmProblem.x[indexI][indexJ+1].value=svmProblem.x[indexJ+1][indexI].value;
        }

      svmModel=svm_train(&svmProblem, &svmParameter);

      for (size_t svId=0; svId<svmModel->l; ++svId){
        size_t index=static_cast<unsigned>(svmModel->SV[svId]->value);
        VectorXd supportVector=trainData.row(index-1);
        supportVectors.insert(std::pair<size_t,VectorXd>(index, std::move(supportVector)));
      }

      modelTrained=true;
      delete[] svmProblem.x;
      delete[] svmProblem.y;
      delete[] vector_x;
    }

    VectorXd predict(const MatrixXd& testData){
      if (!modelTrained) throw std::runtime_error("model has not been trained yet");

      if (testData.cols()!=numberOfFeatures)
        throw std::runtime_error("number of features mismatich");

      size_t numberOfTests=testData.rows();
      VectorXd predictedLabels(numberOfTests);

      libsvm::svm_node* kerVec=new libsvm::svm_node[numberOfTrainData+1];
      for (size_t rowIndex=0; rowIndex<numberOfTests; ++rowIndex){
        for (auto sv : supportVectors){
          kerVec[sv.first].index=sv.first;
          kerVec[sv.first].value=kernel(testData.row(rowIndex), sv.second);
        }
        predictedLabels(rowIndex) = libsvm::svm_predict(svmModel,kerVec);
      }

      delete[] kerVec;
      return predictedLabels;
    }

    void clear(){
      libsvm::svm_free_and_destroy_model(&svmModel);
      svmModel=nullptr;
      supportVectors.clear();
      numberOfTrainData=0;
      modelTrained=false;
    }

    ~LibSVM(){
      libsvm::svm_free_and_destroy_model(&svmModel);
    }

  private:
    const Kernel kernel;
    const size_t numberOfFeatures;
    libsvm::svm_model* svmModel;
    std::map<size_t, VectorXd> supportVectors;
    double C;
    size_t numberOfTrainData;
    double tol;
    double cacheSize=100;
    bool modelTrained=false;
  };

}
