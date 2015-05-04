#ifndef MODEL_LIBSVM
#define MODEL_LIBSVM

#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <external/libsvm.h>
#include <external/libsvm.cpp>

namespace CPPLearn
{
  namespace Models
  {
    /**
     * A Support Vector Classifier based on libsvm implementation.
     */
    template<class Kernel>
    class LibSVM{
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="LibSVM";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::classificationZeroOneLossFunction;

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
      LibSVM(const Kernel kernel_, const size_t numberOfFeatures_,
             const double C_=1.0, const double tol_=1e-5) :
        kernel{kernel_}, numberOfFeatures{numberOfFeatures_}, svmModel{nullptr},
        C{C_}, numberOfTrainData{0}, tol{tol_} { }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels.labelType != ProblemType::Classification){
          throwException("Error happen when training LibSVM model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels.labelData;

        if ((unsigned)trainData.cols() != numberOfFeatures){
          throwException("Error happen when training model, invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
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
        svmProblem.l=(int)numberOfTrainData;
        svmProblem.y=new double[numberOfTrainData];
        svmProblem.x=new libsvm::svm_node*[numberOfTrainData];
        libsvm::svm_node* vector_x=
          new libsvm::svm_node[numberOfTrainData*(numberOfTrainData+2)];

        for (size_t dataIndex=0; dataIndex<numberOfTrainData; ++dataIndex){
          svmProblem.y[dataIndex]=labelData(dataIndex);
          svmProblem.x[dataIndex]=vector_x+(numberOfTrainData+2)*dataIndex;
          svmProblem.x[dataIndex]->index=0;
          svmProblem.x[dataIndex]->value=(double)dataIndex+1;
          (svmProblem.x[dataIndex]+numberOfTrainData+1)->index=-1;
          (svmProblem.x[dataIndex]+numberOfTrainData+1)->value=0;
        }

        for (size_t indexI=0; indexI<numberOfTrainData; ++indexI)
          for (size_t indexJ=0; indexJ<numberOfTrainData; ++indexJ){
            svmProblem.x[indexI][indexJ+1].index=(int)indexJ+1;
            if (indexI <= indexJ){
              Map<const VectorXd> dataI(&trainData(indexI,0),numberOfFeatures);
              Map<const VectorXd> dataJ(&trainData(indexJ,0),numberOfFeatures);
              svmProblem.x[indexI][indexJ+1].value=kernel(dataI,dataJ);
            }
            else
              svmProblem.x[indexI][indexJ+1].value=
                svmProblem.x[indexJ][indexI+1].value;
          }

        svmModel=libsvm::svm_train(&svmProblem, &svmParameter);

        //copy SV into model struct
        libsvm::svm_node* vector_sv =
          (libsvm::svm_node*)malloc(svmModel->l*sizeof(libsvm::svm_node));
        for (size_t svIndex=0; svIndex<(unsigned)svmModel->l; ++svIndex){
          (vector_sv + svIndex)->index=0;
          (vector_sv + svIndex)->value=svmModel->SV[svIndex]->value;
          svmModel->SV[svIndex] = vector_sv + svIndex;
        }

        //hand duty to free SV to svm_model
        svmModel->free_sv=1;

        for (size_t svId=0; svId<(unsigned)svmModel->l; ++svId){
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
      Labels predict(const MatrixXd& testData) const
      {
        if (!modelTrained){
          throwException("Error happen when predicting with LibSVM model: "
                         "Model has not been trained yet!");
        }

        if ((unsigned)testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting with LibSVM: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, testData.cols());
        }

        size_t numberOfTests=testData.rows();
        Labels predictedLabels(ProblemType::Classification);
        predictedLabels.labelData.resize(numberOfTests);

        for (size_t rowIndex=0; rowIndex<numberOfTests; ++rowIndex)
          predictedLabels.labelData(rowIndex) = predictOne(testData.row(rowIndex));

        return predictedLabels;
      }

      double predictOne(const VectorXd& instance) const
      {
        libsvm::svm_node* kerVec=new libsvm::svm_node[numberOfTrainData+2];
        kerVec[numberOfTrainData+1].index=-1;
        kerVec[numberOfTrainData+1].value=0;
        kerVec[0].index=0;

        for (auto& sv : supportVectors){
          kerVec[sv.first].index=(int)sv.first;
          kerVec[sv.first].value=kernel(instance, sv.second);
        }

        double result = libsvm::svm_predict(svmModel,kerVec);

        delete[] kerVec;
        return result;
      }

      /**
       * Each row is a SV.
       */
      const size_t& getNumberOfFeatures() const
      {
        return numberOfFeatures;
      }

      MatrixXd getSupportVectors() const
      {
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
      void clear()
      {
        libsvm::svm_free_and_destroy_model(&svmModel);
        svmModel=nullptr;
        supportVectors.clear();
        numberOfTrainData=0;
        modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return verbose;
      }

      /**
       * Destructor.
       */
      ~LibSVM()
      {
        libsvm::svm_free_and_destroy_model(&svmModel);
      }

    private:
      const Kernel kernel;
      const size_t numberOfFeatures;
      //! SVM model defined in the libsvm library.
      libsvm::svm_model* svmModel;
      //! Support vectors use libsvm index convenstion.
      vector<std::pair<size_t, VectorXd> > supportVectors;
      //! regularization parameter.
      const double C;
      size_t numberOfTrainData;
      double tol;
      //! Indicates whether the model has been trained.
      bool modelTrained=false;
      //! cache size (MB) use by libsvm train.
      const double cacheSize=100;
      VerboseFlag verbose = VerboseFlag::Quiet;
    };
  }
}

#endif //MODEL_LIBSVM
