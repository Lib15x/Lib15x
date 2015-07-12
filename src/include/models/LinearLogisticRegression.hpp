#ifndef MODEL_LINEAR_LOGISTIC_REGRESSION
#define MODEL_LINEAR_LOGISTIC_REGRESSION

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../external/liblinear.h"
#include "../external/liblinear.cpp"
#include "../internal/_BaseClassifier.hpp"

namespace CPPLearn
{
  namespace Models
  {
    class LinearLogisticRegression : public _BaseClassifier<LinearLogisticRegression> {
    public:
      using BaseClassifier = _BaseClassifier<LinearLogisticRegression>;
      using BaseClassifier::train;
      static constexpr const char* ModelName="LinearLogisticRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      LinearLogisticRegression(const long numberOfFeatures, const long numberOfClasses,
                               const Penalty penaltyType, const double C=1.0,
                               const double bias=-1, const double tol=1e-5,
                               const double sparseLevelCutOff=1e-5) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _penaltyType{penaltyType}, _C{C}, _bias{bias}, _tol{tol},
        _sparseLevelCutOff{sparseLevelCutOff} { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        const VectorXd& labelData=trainLabels._labelData;
        assert(weights.size()==labelData.size());
        vector<long> trainIndices;
        for (long dataId=0; dataId<labelData.size(); ++dataId){
          long repeatance=static_cast<long>(weights(dataId));
          if (static_cast<double>(repeatance) != weights(dataId)) {
            throwException("Error happened in LibSVM class: LibSVM cannot handle general "
                           "sample weights=%f", weights(dataId));
          }
          for (long rep=0; rep<repeatance; ++rep)
            trainIndices.push_back(dataId);
        }

        liblinear::parameter linearParam;

        switch (_penaltyType){
        case Penalty::L1 :
          linearParam.solver_type = liblinear::L1R_LR;
          break;
        case Penalty::L2 :
          linearParam.solver_type = liblinear::L2R_LR;
          break;
        default:
          {
            throwException("Error happened when training LinearLogisticRegression model:\n"
                           "the penalty type mush be either L1 or L2!\n");
          }
        }

        linearParam.eps = _tol;
        linearParam.C = _C;
        linearParam.nr_weight = 0;
        linearParam.weight_label = NULL;
        linearParam.weight = NULL;
        linearParam.p = 0.1; //this parameter is not used in logistic regression

        liblinear::problem linearProblem;

        liblinear::feature_node* x_space= nullptr;

        _constructLiblinearProblem(trainData, labelData, trainIndices,
                                   &linearProblem, &x_space);

        _linearModel = liblinear::train(&linearProblem, &linearParam);

        assert(_linearModel->bias==_bias);
        assert(_linearModel->param.solver_type==linearParam.solver_type);
        assert(_linearModel->param.eps==_tol);
        assert(_linearModel->param.C==_C);

        assert(&(_linearModel->param)!=&linearParam);
        assert(_linearModel->nr_feature == BaseClassifier::_numberOfFeatures);

        liblinear::destroy_param(&linearParam);
        free(linearProblem.y);
        free(linearProblem.x);
        free(x_space);
        _modelTrained=true;
      }

      double predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        assert(_linearModel);
        //need to take care of bias term
        long numberOfFeatures = BaseClassifier::_numberOfFeatures;
        long maxNrAttr = 64;
        auto thisNode = (liblinear::feature_node*)
          malloc(maxNrAttr*sizeof(liblinear::feature_node));

        long numberOfNonZeroElementsAndBias=0;
        for (long featId=0; featId<numberOfFeatures; ++featId){
          if (fabs(instance(featId)) > _sparseLevelCutOff){
            if(numberOfNonZeroElementsAndBias >= maxNrAttr-2){  // need one more for index = -1 {
              maxNrAttr *= 2;
              thisNode = (liblinear::feature_node *)
                realloc(thisNode, maxNrAttr*sizeof(liblinear::feature_node));
            }
            thisNode[numberOfNonZeroElementsAndBias].index=static_cast<int>(featId)+1;
            thisNode[numberOfNonZeroElementsAndBias].value=instance(featId);
            ++numberOfNonZeroElementsAndBias;
          }
        }

        //add bias term
        if(_bias>=0) {
          thisNode[numberOfNonZeroElementsAndBias].index = static_cast<int>(numberOfFeatures)+1;
          thisNode[numberOfNonZeroElementsAndBias].value = _bias;
          numberOfNonZeroElementsAndBias++;
        }

        thisNode[numberOfNonZeroElementsAndBias].index = -1;
        double result = liblinear::predict(_linearModel, thisNode);
        free(thisNode);
        return result;
      }

      void _clearModel()
      {
        liblinear::free_and_destroy_model(&_linearModel);
        _linearModel=nullptr;
      }

      LinearLogisticRegression (const LinearLogisticRegression& other) :
        BaseClassifier{other}, _linearModel{nullptr},
        _penaltyType{other._penaltyType},
        _C{other._C}, _bias{other._bias},
        _tol{other._tol}, _sparseLevelCutOff{other._sparseLevelCutOff}
      {
        _linearModel = _copyLibLinearModel(other._linearModel);
      }

      LinearLogisticRegression (LinearLogisticRegression&& other) :
        BaseClassifier{std::move(other)}, _linearModel{nullptr},
        _penaltyType{other._penaltyType},
        _C{other._C}, _bias{other._bias},
        _tol{other._tol}, _sparseLevelCutOff{other._sparseLevelCutOff}
      {
        std::swap(_linearModel, other._linearModel);
      }

      LinearLogisticRegression& operator = (LinearLogisticRegression other)
      {
        BaseClassifier::operator =(other);
        _penaltyType = other._penaltyType;
        _C = other._C;
        _bias = other._bias;
        _tol = other._tol;
        _sparseLevelCutOff = other._sparseLevelCutOff;
        std::swap(_linearModel, other._linearModel);
        return *this;
      }

      LinearLogisticRegression& operator = (LinearLogisticRegression&& other)
      {
        BaseClassifier::operator =(std::move(other));
        _penaltyType = other._penaltyType;
        _C = other._C;
        _bias = other._bias;
        _tol = other._tol;
        _sparseLevelCutOff = other._sparseLevelCutOff;
        std::swap(_linearModel, other._linearModel);
        return *this;
      }

      ~LinearLogisticRegression()
      {
        liblinear::free_and_destroy_model(&_linearModel);
      }
    private:
      liblinear::model* _copyLibLinearModel(const liblinear::model* const other)
      {
        if (!other) return nullptr;
        liblinear::model *copy = (liblinear::model *)malloc(sizeof(liblinear::model));

        copy->param = other->param;
        copy->nr_class = other->nr_class;
        copy->nr_feature = other->nr_feature;
        copy->w = nullptr;
        copy->label = nullptr;
        copy->bias = other->bias;

        if (other->label) {
          copy->label = (int*)malloc(other->nr_class*sizeof(int));
          for (int i=0; i<other->nr_class; ++i)
            copy->label[i] = other->label[i];
        }

        if (other->label) {
          copy->label = (int*)malloc(other->nr_class*sizeof(int));
          for (int i=0; i<other->nr_class; ++i)
            copy->label[i] = other->label[i];
        }

        if (other->w) {
          int n;
          int nr_feature=other->nr_feature;
          if(other->bias>=0)
            n=nr_feature+1;
          else
            n=nr_feature;

          int w_size = n;
          int nr_w;
          if(other->nr_class==2 && other->param.solver_type != liblinear::MCSVM_CS)
            nr_w = 1;
          else
            nr_w = other->nr_class;

          copy->w=(double*)malloc(w_size*nr_w*sizeof(double));
          for(int i=0; i<w_size*nr_w; i++) {
                copy->w[i]=other->w[i];
          }
        }

        return copy;
      }

    private:
      void _constructLiblinearProblem(const MatrixXd& trainData, const VectorXd& labelData,
                                      const vector<long>& trainIndices,
                                      liblinear::problem* prob,
                                      liblinear::feature_node** x_space) const
      {
        long numberOfFeatures = BaseClassifier::_numberOfFeatures;
        long numberOfData=trainIndices.size();
        vector<long> nonZeroIndexX;
        nonZeroIndexX.reserve(numberOfData*numberOfFeatures);
        vector<long> nonZeroIndexY;
        nonZeroIndexY.reserve(numberOfData*numberOfFeatures);
        vector<long> nonZeroFeatureCountEachData(numberOfData, 0);

        prob->l = static_cast<int>(numberOfData);
        long numberOfNonZeroElementsAndBias=0;

        //for (auto& sampleId : trainIndices) {
        for (long sampleId=0; sampleId<numberOfData; ++sampleId){
          long dataIndex = trainIndices[sampleId];
          for (long featId=0; featId<numberOfFeatures; ++featId)
            if (fabs(trainData(dataIndex,featId))>_sparseLevelCutOff) {
              nonZeroIndexX.push_back(sampleId);
              nonZeroIndexY.push_back(featId);
              ++nonZeroFeatureCountEachData[sampleId];
              ++numberOfNonZeroElementsAndBias;
            }
          ++numberOfNonZeroElementsAndBias;
        }

        prob->bias=_bias;
        prob->y = (double*) malloc(prob->l*sizeof(double));
        prob->x = (liblinear::feature_node**) malloc(prob->l*sizeof(liblinear::feature_node*));
        *x_space = (liblinear::feature_node*)
          malloc((numberOfNonZeroElementsAndBias+prob->l)*sizeof(liblinear::feature_node));

        long currentPosInXSpace=0;
        long currentPosInData=0;

        for(long sampleId=0; sampleId<numberOfData; ++sampleId) {
          long dataIndex=trainIndices[sampleId];
          //the first two operator cannt collapes
          prob->x[sampleId] = &(*x_space)[currentPosInXSpace];
          prob->y[sampleId] = labelData[dataIndex];
          for (long nonZeroId=0; nonZeroId<nonZeroFeatureCountEachData[sampleId]; ++nonZeroId) {
            long xIndex=nonZeroIndexX[currentPosInData];
            long yIndex=nonZeroIndexY[currentPosInData];
            (*x_space)[currentPosInXSpace].index = static_cast<int>(yIndex)+1;
            (*x_space)[currentPosInXSpace].value=trainData(xIndex,yIndex);
            ++currentPosInXSpace;
            ++currentPosInData;
          }

          if(_bias >= 0)
            (*x_space)[currentPosInXSpace++].value = _bias;
          (*x_space)[currentPosInXSpace++].index = -1;
        }

        if(prob->bias >= 0) {
          prob->n = static_cast<int>(numberOfFeatures)+1;
          for(long dataId=1; dataId < prob->l; ++dataId)
            (prob->x[dataId]-2)->index = prob->n;
          (*x_space)[currentPosInXSpace-2].index = prob->n;
        }
        else
          prob->n = static_cast<int>(numberOfFeatures);
      }

    private:
      liblinear::model* _linearModel;
      Penalty _penaltyType;
      double _C;
      double _bias;
      double _tol;
      double _sparseLevelCutOff;
    };
  }
}

#endif //MODEL_LINEAR_LOGISTIC_REGRESSION
