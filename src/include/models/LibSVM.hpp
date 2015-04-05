#ifndef MODEL_LIBSVM
#define MODEL_LIBSVM

#include <core/Definitions.hpp>
#include <external/svm.h>
#include <external/svm.cpp>

namespace CPPLearn{
  namespace Models{

    /**
     * A Support Vector Classifier based on libsvm implementation.
     */
    template<class Kernel>
    class LibSVM{
    public:
      /**
       * Creates the model, with empty model initialized.
       *
       * @param kernel_ kernel function object used for transformation, typical
       * types include linear, RBF or sigmoid.
       * @param numberOfFeatures_ number of features of the model,
       * required user provided befor hand.
       * @param C_ regularization constant.
       * @param tol_ stopping critiria.
       */
      LibSVM(Kernel kernel_, size_t numberOfFeatures_,
             double C_=1.0, double tol_=1e-5) :
        kernel{kernel_}, numberOfFeatures{numberOfFeatures_}, svmModel{nullptr},
        C{C_}, numberOfTrainData{0}, tol{tol_} { }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void train(const MatrixXd& trainData, const VectorXd& trainLabels) {
        if (trainData.cols() != numberOfFeatures){
          throwException("Error happen when training model, invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%lu).\n",
                         numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != trainLabels.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%lu), "
                         "number of labels: (%lu), ",
                         trainData.rows(), trainLabels.size());
        }

        numberOfTrainData = trainData.rows();

        libsvm::svm_parameter svmParameter;
        svmParameter.svm_type=libsvm::C_SVC;
        svmParameter.kernel_type=libsvm::PRECOMPUTED;
        svmParameter.C=C;
        svmParameter.eps=tol;
        svmParameter.cache_size=cacheSize;
        svmParameter.nr_weight = 0;
        svmParameter.weight_label = NULL;
        svmParameter.weight = NULL;
        svmParameter.shrinking = 1;
        svmParameter.probability = 0;

        libsvm::svm_problem svmProblem;
        svmProblem.l=numberOfTrainData;
        svmProblem.y=new double[numberOfTrainData];
        svmProblem.x=new libsvm::svm_node*[numberOfTrainData];
        libsvm::svm_node* vector_x=
          new libsvm::svm_node[numberOfTrainData*(numberOfTrainData+2)];

        for (size_t dataIndex=0; dataIndex<numberOfTrainData; ++dataIndex){
          svmProblem.y[dataIndex]=trainLabels(dataIndex);
          svmProblem.x[dataIndex]=vector_x+(numberOfTrainData+2)*dataIndex;
          svmProblem.x[dataIndex]->index=0;
          svmProblem.x[dataIndex]->value=dataIndex+1;
          (svmProblem.x[dataIndex]+numberOfTrainData+1)->index=-1;
          (svmProblem.x[dataIndex]+numberOfTrainData+1)->value=0;
        }

        for (size_t indexI=0; indexI<numberOfTrainData; ++indexI)
          for (size_t indexJ=0; indexJ<numberOfTrainData; ++indexJ){
            svmProblem.x[indexI][indexJ+1].index=indexJ+1;
            if (indexI <= indexJ)
              svmProblem.x[indexI][indexJ+1].value=kernel(trainData.row(indexI),
                                                          trainData.row(indexJ));
            else
              svmProblem.x[indexI][indexJ+1].value=
                svmProblem.x[indexJ][indexI+1].value;
          }

        svmModel=svm_train(&svmProblem, &svmParameter);

        //copy SV into model struct
        libsvm::svm_node* vector_sv = new libsvm::svm_node[svmModel->l];
        for (size_t svIndex=0; svIndex<svmModel->l; ++svIndex){
          (vector_sv + svIndex)->index=0;
          (vector_sv + svIndex)->value=svmModel->SV[svIndex]->value;
          svmModel->SV[svIndex] = vector_sv + svIndex;
        }

        //hand duty to free SV to svm_model
        svmModel->free_sv=1;

        for (size_t svId=0; svId<svmModel->l; ++svId){
          size_t index=static_cast<unsigned>(svmModel->SV[svId]->value);
          VectorXd supportVector=trainData.row(index-1);
          supportVectors.push_back(std::pair<size_t,VectorXd> (index, std::move(supportVector)));
        }

        modelTrained=true;
        delete[] svmProblem.x;
        delete[] svmProblem.y;
        delete[] vector_x;
      }

      /**
       * Calculate predictions based on trained model, returns the predicted
       * labels. The model has to be trained first.
       *
       * @param testData predictors, the number of columns should be the
       * same as number of features.
       */
      VectorXd predict(const MatrixXd& testData) const {
        if (!modelTrained){
          throwException("model has not been trained yet!");
        }

        if (testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting, invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%lu).\n",
                         numberOfFeatures, testData.cols());
        }

        size_t numberOfTests=testData.rows();
        VectorXd predictedLabels(numberOfTests);

        libsvm::svm_node* kerVec=new libsvm::svm_node[numberOfTrainData+2];
        kerVec[numberOfTrainData+1].index=-1;
        kerVec[numberOfTrainData+1].value=0;
        kerVec[0].index=0;
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

      /**
       * Each row is a SV.
       */
      MatrixXd getSupportVectors() const {
        if (!modelTrained){
          throwException("model has not been trained yet!");
        }

        size_t numberOfSVs=supportVectors.size();

        MatrixXd supportVectorData(numberOfSVs, numberOfFeatures);
        for (size_t svIndex=0; svIndex<numberOfSVs; ++svIndex)
          supportVectorData.row(svIndex)=supportVectors[svIndex].second;

        return supportVectorData;
      };

      /**
       * Clear the model.
       */
      void clear(){
        libsvm::svm_free_and_destroy_model(&svmModel);
        svmModel=nullptr;
        supportVectors.clear();
        numberOfTrainData=0;
        modelTrained=false;
      }

      /**
       * Destructor.
       */
      ~LibSVM(){
        libsvm::svm_free_and_destroy_model(&svmModel);
      }

    private:
      const Kernel kernel;
      const size_t numberOfFeatures;
      //! SVM model defined in the libsvm library.
      libsvm::svm_model* svmModel;
      //! Support vectors use libsvm index convenstion.
      vector<std::pair<size_t, VectorXd>> supportVectors;
      //! regularization parameter.
      double C;
      size_t numberOfTrainData;
      double tol;
      //! Indicates whether the model has been trained.
      bool modelTrained=false;
      //! cache size (MB) use by libsvm train.
      double cacheSize=100;
    };
  }
}

#endif //MODEL_LIBSVM
