#ifndef MODEL_LIBSVM
#define MODEL_LIBSVM

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../external/libsvm.h"
#include "../external/libsvm.cpp"
#include "../internal/_BaseClassifier.hpp"

namespace Lib15x
{
  namespace Models
  {
    /**
     * A Support Vector Classifier based on libsvm implementation.
     */
    template<class Kernel>
    class LibSVM : public _BaseClassifier<LibSVM<Kernel> > {
    public:
      using BaseClassifier = _BaseClassifier<LibSVM<Kernel> >;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "LibSVM";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      LibSVM(const long numberOfFeatures, const long numberOfClasses,
             const double C, const Kernel kernel) :
        BaseClassifier{numberOfFeatures, numberOfClasses}, _svmModel{nullptr},
        _C{C}, _numberOfTrainData{0}, _kernel{kernel} { }

      LibSVM(const long numberOfFeatures, const long numberOfClasses,
             const Kernel kernel) :
        LibSVM{numberOfFeatures, numberOfClasses, 1.0, Kernel{kernel} } { }

      template<typename... Args>
      LibSVM(const long numberOfFeatures, const long numberOfClasses,
             const double C, const Args... args) :
        LibSVM{numberOfFeatures, numberOfClasses, C, Kernel{args...} } { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        const VectorXd& labelData=trainLabels._labelData;
        assert(weights.size()==labelData.size());
        vector<long> trainIndices;
        for (long dataId=0; dataId<labelData.size(); ++dataId){
          long dataRepeat=static_cast<long>(weights(dataId));
          if (static_cast<double>(dataRepeat) != weights(dataId)) {
            throwException("Error happened in LibSVM class: LibSVM cannot handle general "
                           "sample weights=%f", weights(dataId));
          }
          for (long rep=0; rep<dataRepeat; ++rep)
            trainIndices.push_back(dataId);
        }
        _numberOfTrainData = trainIndices.size();

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

        for (long sampleId=0; sampleId<_numberOfTrainData; ++sampleId){
          const long dataIndex=trainIndices[sampleId];
          svmProblem.y[sampleId]=labelData(dataIndex);
          svmProblem.x[sampleId]=vector_x+(_numberOfTrainData+2)*sampleId;
          svmProblem.x[sampleId]->index=0;
          svmProblem.x[sampleId]->value=static_cast<double>(sampleId)+1.0;
          (svmProblem.x[sampleId]+_numberOfTrainData+1)->index=-1;
          (svmProblem.x[sampleId]+_numberOfTrainData+1)->value=0;
        }

        for (long sampleIdI=0; sampleIdI<_numberOfTrainData; ++sampleIdI){
          const long dataIdI=trainIndices[sampleIdI];
          for (long sampleIdJ=0; sampleIdJ<_numberOfTrainData; ++sampleIdJ){
            svmProblem.x[sampleIdI][sampleIdJ+1].index=static_cast<int>(sampleIdJ)+1;
            const long dataIdJ=trainIndices[sampleIdJ];
            if (sampleIdI <= sampleIdJ){
              Map<const VectorXd> dataI(&trainData(dataIdI,0), BaseClassifier::_numberOfFeatures);
              Map<const VectorXd> dataJ(&trainData(dataIdJ,0), BaseClassifier::_numberOfFeatures);
              svmProblem.x[sampleIdI][sampleIdJ+1].value=_kernel(dataI,dataJ);
            }
            else
              svmProblem.x[sampleIdI][sampleIdJ+1].value=
                svmProblem.x[sampleIdJ][sampleIdI+1].value;
          }
        }

        _svmModel=libsvm::svm_train(&svmProblem, &svmParameter);

        //copy SV into model struct
        auto vector_sv = (libsvm::svm_node*)malloc(_svmModel->l*sizeof(libsvm::svm_node));
        for (long svIndex=0; svIndex<_svmModel->l; ++svIndex){
          (vector_sv + svIndex)->index=0;
          (vector_sv + svIndex)->value=_svmModel->SV[svIndex]->value;
          _svmModel->SV[svIndex] = vector_sv + svIndex;
        }

        //hand duty to free SV to svm_model
        _svmModel->free_sv=1;

        for (long svId=0; svId<_svmModel->l; ++svId){
          const long index=static_cast<long>(_svmModel->SV[svId]->value);
          _supportVectors.push_back(std::make_pair(index, trainData.row(trainIndices[index-1])));
        }

        delete[] svmProblem.x;
        delete[] svmProblem.y;
        delete[] vector_x;

        BaseClassifier::_modelTrained=true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        assert(_svmModel);

        auto kerVec=new libsvm::svm_node[_numberOfTrainData+2];
        kerVec[_numberOfTrainData+1].index=-1;
        kerVec[_numberOfTrainData+1].value=0;
        kerVec[0].index=0;

        for (const auto& sv : _supportVectors){
          kerVec[sv.first].index = static_cast<int>(sv.first);
          kerVec[sv.first].value = _kernel(instance, sv.second);
        }

        double result = libsvm::svm_predict(_svmModel,kerVec);
        delete[] kerVec;

        return result;
      }

      MatrixXd getSupportVectors() const
      {
        if (!BaseClassifier::_modelTrained) {
          throwException("%s has not been trained yet!\n", ModelName);
        }

        const long numberOfSVs=static_cast<long>(_supportVectors.size());
        MatrixXd supportVectorData{numberOfSVs, BaseClassifier::_numberOfFeatures};

        for (long svIndex=0; svIndex<numberOfSVs; ++svIndex)
          supportVectorData.row(svIndex)=_supportVectors[svIndex].second;

        return supportVectorData;
      };

      void _clearModel()
      {
        libsvm::svm_free_and_destroy_model(&_svmModel);
        _svmModel=nullptr;
        _supportVectors.clear();
        _numberOfTrainData=0;
      }

      double& setTolerence(){
        return _tol;
      }

      LibSVM (const LibSVM& other) :
        BaseClassifier{other}, _svmModel{nullptr},
        _supportVectors{other._supportVectors},
        _C{other._C}, _numberOfTrainData{other._numberOfTrainData},
        _tol{other._tol}, _cacheSize{other._cacheSize},
        _kernel{other._kernel}
      {
        _svmModel = _copyLibSVMModel(other._svmModel);
      }

      LibSVM (LibSVM&& other) :
        BaseClassifier{std::move(other)}, _svmModel{nullptr},
        _supportVectors{std::move(other._supportVectors)},
        _C{other._C}, _numberOfTrainData{other._numberOfTrainData},
        _tol{other._tol}, _cacheSize{other._cacheSize},
        _kernel{std::move(other._kernel)}
      {
        std::swap(_svmModel, other._svmModel);
      }

      LibSVM& operator = (LibSVM other)
      {
        BaseClassifier::operator =(other);
        _C = other._C;
        _numberOfTrainData = other._numberOfTrainData;
        _tol = other._tol;
        _cacheSize= other._cacheSize;
        _kernel = other._kernel;
        _supportVectors=std::move(other._supportVectors);
        std::swap(_svmModel, other._svmModel);
        return *this;
      }

      LibSVM& operator = (LibSVM&& other)
      {
        BaseClassifier::operator =(std::move(other));
        _C = other._C;
        _numberOfTrainData = other._numberOfTrainData;
        _tol = other._tol;
        _cacheSize= other._cacheSize;
        _supportVectors=std::move(other._supportVectors);
        std::swap(_svmModel, other._svmModel);
        _kernel = std::move(other._kernel);
        return *this;
      }

      /**
       * Destructor.
       */
      ~LibSVM()
      {
        libsvm::svm_free_and_destroy_model(&_svmModel);
      }

    private:
      libsvm::svm_model* _copyLibSVMModel(const libsvm::svm_model* const svmModel)
      {
        if (!svmModel) return nullptr;

        libsvm::svm_model *model = (libsvm::svm_model *)malloc(sizeof(libsvm::svm_model));
        model->param = svmModel->param;
        model-> nr_class = svmModel->nr_class;
        model-> l = svmModel->l;
        model->free_sv = svmModel->free_sv;

        model->rho = nullptr;
        model->probA = nullptr;
        model->probB = nullptr;
        model->sv_indices = nullptr;
        model->label = nullptr;
        model->nSV = nullptr;
        model->SV = nullptr;
        model->sv_coef = nullptr;

        if (svmModel->rho) {
          int n = model->nr_class * (model->nr_class-1)/2;
          model->rho = (double*)malloc(n*sizeof(double));
          for(int i=0;i<n;i++) model->rho[i] = svmModel->rho[i];
        }

        if (svmModel->label) {
          int n = model->nr_class;
          model->label = (int*)malloc(n*sizeof(int));
          for(int i=0;i<n;i++) model->label[i] = svmModel->label[i];
        }

        if (svmModel->probA) {
          int n = model->nr_class * (model->nr_class-1)/2;
          model->probA = (double*)malloc(n*sizeof(double));
          for(int i=0;i<n;i++) model->probA[i] = svmModel->probA[i];
        }

        if (svmModel->probB) {
          int n = model->nr_class * (model->nr_class-1)/2;
          model->probB = (double*)malloc(n*sizeof(double));
          for(int i=0;i<n;i++) model->probB[i] = svmModel->probB[i];
        }

        if (svmModel->nSV) {
          int n = model->nr_class;
          model->nSV = (int*)malloc(n*sizeof(int));
          for(int i=0;i<n;i++) model->nSV[i] = svmModel->nSV[i];
        }

        if (svmModel->sv_indices) {
          int n = model->l;
          model->sv_indices = (int*)malloc(n*sizeof(int));
          for(int i=0;i<n;i++) model->sv_indices[i] = svmModel->sv_indices[i];
        }

        if (svmModel->SV) {
          model->SV = (libsvm::svm_node**)malloc(svmModel->l*sizeof(libsvm::svm_node*));
          auto vector_sv = (libsvm::svm_node*)malloc(svmModel->l*sizeof(libsvm::svm_node));
          for (long svIndex = 0; svIndex < svmModel->l; ++svIndex){
            (vector_sv + svIndex)->index=0;
            (vector_sv + svIndex)->value=svmModel->SV[svIndex]->value;
            model->SV[svIndex] = vector_sv + svIndex;
          }
        }

        if (svmModel->sv_coef) {
          int m = model->nr_class - 1;
          model->sv_coef = (double**)malloc(m*sizeof(double *));
          for(int i=0;i<m;i++) {
            model->sv_coef[i] = (double*)malloc(svmModel->l*sizeof(double));
            for (int j=0; j<svmModel->l;j++)
              model->sv_coef[i][j]=svmModel->sv_coef[i][j];
          }
        }

        return model;
      }

    private:
      //! SVM model defined in the libsvm library.
      libsvm::svm_model* _svmModel;
      //! Support vectors use libsvm index convenstion.
      vector<std::pair<long, VectorXd> > _supportVectors;
      //! regularization parameter.
      double _C;
      long _numberOfTrainData;
      double _tol=1e-9;
      //! cache size (MB) use by libsvm train.
      double _cacheSize=100;
      Kernel _kernel;
    };
  }
}

#endif //MODEL_LIBSVM
