#ifndef MODEL_LINEAR_LOGISTIC_REGRESSION
#define MODEL_LINEAR_LOGISTIC_REGRESSION

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../external/liblinear.h"
#include "../external/liblinear.cpp"

namespace CPPLearn
{
  namespace Models
  {
    class LinearLogisticRegression {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="LinearLogisticRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::classificationZeroOneLossFunction;

      LinearLogisticRegression(const long numberOfFeatures, const Penalty penaltyType,
                               const double C=1.0, const double bias=-1,
                               const double tol=1e-5, const double sparseLevelCutOff=1e-5) :
        _numberOfFeatures{numberOfFeatures}, _linearModel{nullptr},
        _penaltyType{penaltyType}, _C{C}, _bias{bias}, _tol{tol},
        _sparseLevelCutOff{sparseLevelCutOff} { }

      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Classification){
          throwException("Error happen when training LinearLogisticRegression model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training LinearLogisticRegression model, "
                         "invalid inpute data: "
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

        constructLiblinearProblem(trainData, labelData, &linearProblem, &x_space);

        _linearModel = liblinear::train(&linearProblem, &linearParam);

        assert(_linearModel->bias==_bias);
        assert(_linearModel->param.solver_type==linearParam.solver_type);
        assert(_linearModel->param.eps==_tol);
        assert(_linearModel->param.C==_C);

        assert(&(_linearModel->param)!=&linearParam);
        assert(_linearModel->nr_feature == _numberOfFeatures);

        liblinear::destroy_param(&linearParam);
        free(linearProblem.y);
        free(linearProblem.x);
        free(x_space);
        _modelTrained=true;
      }

      Labels predict(const MatrixXd& testData) const
      {
        if (!_modelTrained){
          throwException("Error happen when predicting with LinearLogisticRegression model: "
                         "Model has not been trained yet!");
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with LinearLogisticRegression: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, testData.cols());
        }

        long numberOfTests=testData.rows();
        Labels predictedLabels{ProblemType::Classification};
        predictedLabels._labelData.resize(numberOfTests);

        for (long rowIndex=0; rowIndex<numberOfTests; ++rowIndex){
          Map<const VectorXd> instance(&testData(rowIndex, 0), _numberOfFeatures);
          predictedLabels._labelData(rowIndex) = predictOne(instance);
        }

        return predictedLabels;
      }

      double predictOne(const VectorXd& instance) const
      {
        //need to take care of bias term
        long maxNrAttr = 64;
        auto thisNode = (liblinear::feature_node*)
          malloc(maxNrAttr*sizeof(liblinear::feature_node));

        long numberOfNonZeroElementsAndBias=0;
        for (long featId=0l; featId<_numberOfFeatures; ++featId){
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
          thisNode[numberOfNonZeroElementsAndBias].index = static_cast<int>(_numberOfFeatures)+1;
          thisNode[numberOfNonZeroElementsAndBias].value = _bias;
          numberOfNonZeroElementsAndBias++;
        }

        thisNode[numberOfNonZeroElementsAndBias].index = -1;
        double result = liblinear::predict(_linearModel, thisNode);
        free(thisNode);
        return result;
      }

      const long& getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }


      /**
       * Clear the model.
       */
      void clear()
      {
        liblinear::free_and_destroy_model(&_linearModel);
        _linearModel=nullptr;
        _modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return _verbose;
      }

      /**
       * Destructor.
       */
      ~LinearLogisticRegression()
      {
        liblinear::free_and_destroy_model(&_linearModel);
      }

    private:
      const long _numberOfFeatures;
      //! liblinear model defined in the liblinear library.
      liblinear::model* _linearModel;
      const Penalty _penaltyType;
      //! regularization parameter.
      const double _C;
      const double _bias;
      double _tol;
      //! Indicates whether the model has been trained.
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
      const double _sparseLevelCutOff;

    private:
      void constructLiblinearProblem(const MatrixXd& trainData, const VectorXd& labelData,
                                     liblinear::problem* prob,
                                     liblinear::feature_node** x_space) const
      {
        long numberOfData=trainData.rows();
        vector<long> nonZeroIndexX;
        nonZeroIndexX.reserve(numberOfData*_numberOfFeatures);
        vector<long> nonZeroIndexY;
        nonZeroIndexY.reserve(numberOfData*_numberOfFeatures);
        vector<long> nonZeroFeatureCountEachData(numberOfData, 0);

        prob->l = static_cast<int>(numberOfData);
        long numberOfNonZeroElementsAndBias=0;

        for (long dataId=0; dataId<numberOfData; ++dataId){
          for (long featId=0; featId<_numberOfFeatures; ++featId)
            if (fabs(trainData(dataId,featId))>_sparseLevelCutOff) {
              nonZeroIndexX.push_back(dataId);
              nonZeroIndexY.push_back(featId);
              ++nonZeroFeatureCountEachData[dataId];
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

        for(long dataId=0; dataId<numberOfData; ++dataId) {
          prob->x[dataId] = &(*x_space)[currentPosInXSpace]; //the first two operator cannt collapes
          prob->y[dataId] = labelData[dataId];
          for (long nonZeroId=0; nonZeroId<nonZeroFeatureCountEachData[dataId]; ++nonZeroId){
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
          prob->n = static_cast<int>(_numberOfFeatures)+1;
          for(long dataId=1; dataId < prob->l; ++dataId)
            (prob->x[dataId]-2)->index = prob->n;
          (*x_space)[currentPosInXSpace-2].index = prob->n;
        }
        else
          prob->n = static_cast<int>(_numberOfFeatures);
      }
    };
  }
}

#endif //MODEL_LINEAR_LOGISTIC_REGRESSION
