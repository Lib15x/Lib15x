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
      LibSVM(const Kernel kernel, const long numberOfFeatures,
             const double C=1.0, const double tol=1e-5) :
        _kernel{std::move(kernel)}, _numberOfFeatures{numberOfFeatures},
        _svmModel{nullptr}, _C{C}, _numberOfTrainData{0}, _tol{tol} { }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Classification){
          throwException("Error happen when training LibSVM model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training model, invalid inpute data: "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }

        _numberOfTrainData = trainData.rows();

        libsvm::svm_parameter svmParameter;
        svmParameter.svm_type=libsvm::C_SVC;
        svmParameter.kernel_type=libsvm::PRECOMPUTED;
        svmParameter.C=_C;
        svmParameter.eps=_tol;
        svmParameter.cache_size=_cacheSize;
        svmParameter.nr_weight = 0;
        svmParameter.weight_label = NULL;
        svmParameter.weight = NULL;
        svmParameter.shrinking = 1;
        svmParameter.probability = 0;

        libsvm::svm_problem svmProblem;
        svmProblem.l = static_cast<int>(_numberOfTrainData);
        svmProblem.y=new double[_numberOfTrainData];
        svmProblem.x=new libsvm::svm_node*[_numberOfTrainData];
        auto vector_x= new libsvm::svm_node[_numberOfTrainData*(_numberOfTrainData+2)];

        for (long dataIndex=0; dataIndex<_numberOfTrainData; ++dataIndex){
          svmProblem.y[dataIndex]=labelData(dataIndex);
          svmProblem.x[dataIndex]=vector_x+(_numberOfTrainData+2)*dataIndex;
          svmProblem.x[dataIndex]->index=0;
          svmProblem.x[dataIndex]->value=static_cast<double>(dataIndex)+1.0;
          (svmProblem.x[dataIndex]+_numberOfTrainData+1)->index=-1;
          (svmProblem.x[dataIndex]+_numberOfTrainData+1)->value=0;
        }

        for (long indexI=0; indexI<_numberOfTrainData; ++indexI)
          for (long indexJ=0; indexJ<_numberOfTrainData; ++indexJ){
            svmProblem.x[indexI][indexJ+1].index=static_cast<int>(indexJ)+1;
            if (indexI <= indexJ){
              Map<const VectorXd> dataI(&trainData(indexI,0),_numberOfFeatures);
              Map<const VectorXd> dataJ(&trainData(indexJ,0),_numberOfFeatures);
              svmProblem.x[indexI][indexJ+1].value=_kernel(dataI,dataJ);
            }
            else
              svmProblem.x[indexI][indexJ+1].value=
                svmProblem.x[indexJ][indexI+1].value;
          }

        _svmModel=libsvm::svm_train(&svmProblem, &svmParameter);

        //copy SV into model struct
        auto vector_sv = (libsvm::svm_node*)malloc(_svmModel->l*sizeof(libsvm::svm_node));
        for (long svIndex=0l; svIndex<_svmModel->l; ++svIndex){
          (vector_sv + svIndex)->index=0;
          (vector_sv + svIndex)->value=_svmModel->SV[svIndex]->value;
          _svmModel->SV[svIndex] = vector_sv + svIndex;
        }

        //hand duty to free SV to svm_model
        _svmModel->free_sv=1;

        for (long svId=0l; svId<_svmModel->l; ++svId){
          const long& index=static_cast<long>(_svmModel->SV[svId]->value);
          _supportVectors.push_back(std::make_pair(index, trainData.row(index-1)));
        }

        _modelTrained=true;
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
        if (!_modelTrained){
          throwException("Error happen when predicting with LibSVM model: "
                         "Model has not been trained yet!");
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with LibSVM: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, testData.cols());
        }

        const long numberOfTests=testData.rows();
        Labels predictedLabels{ProblemType::Classification};
        predictedLabels._labelData.resize(numberOfTests);

        for (long rowIndex=0; rowIndex<numberOfTests; ++rowIndex){
          Map<const VectorXd> instance{&testData(rowIndex, 0), _numberOfFeatures};
          predictedLabels._labelData(rowIndex) = predictOne(instance);
        }

        return predictedLabels;
      }

      double predictOne(const VectorXd& instance) const
      {
        auto kerVec=new libsvm::svm_node[_numberOfTrainData+2];
        kerVec[_numberOfTrainData+1].index=-1;
        kerVec[_numberOfTrainData+1].value=0;
        kerVec[0].index=0;

        for (const auto& sv : _supportVectors){
          kerVec[sv.first].index=static_cast<int>(sv.first);
          kerVec[sv.first].value=_kernel(instance, sv.second);
        }

        double result = libsvm::svm_predict(_svmModel,kerVec);
        delete[] kerVec;

        return result;
      }

      /**
       * Each row is a SV.
       */
      const long& getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }

      MatrixXd getSupportVectors() const
      {
        if (!_modelTrained){
          throwException("model has not been trained yet!");
        }

        const long numberOfSVs=static_cast<long>(_supportVectors.size());

        MatrixXd supportVectorData{numberOfSVs, _numberOfFeatures};
        for (long svIndex=0; svIndex<numberOfSVs; ++svIndex)
          supportVectorData.row(svIndex)=_supportVectors[svIndex].second;

        return supportVectorData;
      };

      /**
       * Clear the model.
       */
      void clear()
      {
        libsvm::svm_free_and_destroy_model(&_svmModel);
        _svmModel=nullptr;
        _supportVectors.clear();
        _numberOfTrainData=0;
        _modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return _verbose;
      }

      /**
       * Destructor.
       */
      ~LibSVM()
      {
        libsvm::svm_free_and_destroy_model(&_svmModel);
      }

    private:
      const Kernel _kernel;
      const long _numberOfFeatures;
      //! SVM model defined in the libsvm library.
      libsvm::svm_model* _svmModel;
      //! Support vectors use libsvm index convenstion.
      vector<std::pair<long, VectorXd> > _supportVectors;
      //! regularization parameter.
      const double _C;
      long _numberOfTrainData;
      double _tol;
      //! Indicates whether the model has been trained.
      bool _modelTrained=false;
      //! cache size (MB) use by libsvm train.
      const double _cacheSize=100;
      VerboseFlag _verbose = VerboseFlag::Quiet;
    };
  }
}

#endif //MODEL_LIBSVM
