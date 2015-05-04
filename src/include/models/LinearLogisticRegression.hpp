#ifndef MODEL_LINEAR_LOGISTIC_REGRESSION
#define MODEL_LINEAR_LOGISTIC_REGRESSION

#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <external/liblinear.h>
#include <external/liblinear.cpp>

namespace CPPLearn
{
  namespace Models
  {
    class LinearLogisticRegression{
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="LinearLogisticRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::classificationZeroOneLossFunction;

      LinearLogisticRegression(const size_t numberOfFeatures_, const Penalty penaltyType_,
                               const double C_=1.0, const double bias_=-1,
                               const double tol_=1e-5, const double sparseLevelCutOff_=1e-5) :
        numberOfFeatures{numberOfFeatures_}, linearModel{nullptr},
        penaltyType{penaltyType_}, C{C_}, bias{bias_}, tol{tol_},
        sparseLevelCutOff{sparseLevelCutOff_} { }

      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels.labelType != ProblemType::Classification){
          throwException("Error happen when training LinearLogisticRegression model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels.labelData;

        if ((unsigned)trainData.cols() != numberOfFeatures){
          throwException("Error happen when training LinearLogisticRegression model, "
                         "invalid inpute data: "
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

        liblinear::parameter linearParam;

        switch (penaltyType){
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

        linearParam.eps = tol;
        linearParam.C = C;
        linearParam.nr_weight = 0;
        linearParam.weight_label = NULL;
        linearParam.weight = NULL;
        linearParam.p = 0.1; //this parameter is not used in logistic regression

        liblinear::problem linearProblem;

        liblinear::feature_node* x_space= nullptr;

        constructLiblinearProblem(trainData, labelData, &linearProblem,
                                  &x_space, sparseLevelCutOff);

        linearModel = liblinear::train(&linearProblem, &linearParam);

        assert(linearModel->bias==bias);
        assert(linearModel->param.solver_type==linearParam.solver_type);
        assert(linearModel->param.eps==tol);
        assert(linearModel->param.C==C);

        assert(&(linearModel->param)!=&linearParam);
        assert(linearModel->nr_feature == (int)numberOfFeatures);

        liblinear::destroy_param(&linearParam);
        free(linearProblem.y);
        free(linearProblem.x);
        free(x_space);
        modelTrained=true;
      }

      Labels predict(const MatrixXd& testData) const
      {
        if (!modelTrained){
          throwException("Error happen when predicting with LinearLogisticRegression model: "
                         "Model has not been trained yet!");
        }

        if ((unsigned)testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting with LinearLogisticRegression: "
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
        //need to take care of bias term
        size_t maxNrAttr = 64;
        liblinear::feature_node* thisNode=
          (liblinear::feature_node *) malloc(maxNrAttr*sizeof(liblinear::feature_node));

        size_t numberOfNonZeroElementsAndBias=0;
        for (size_t featId=0; featId<numberOfFeatures; ++featId){
          if (fabs(instance(featId)) > sparseLevelCutOff){
            if(numberOfNonZeroElementsAndBias>=maxNrAttr-2){  // need one more for index = -1 {
              maxNrAttr *= 2;
              thisNode = (liblinear::feature_node *)
                realloc(thisNode, maxNrAttr*sizeof(liblinear::feature_node));
            }
            thisNode[numberOfNonZeroElementsAndBias].index=(int)featId+1;
            thisNode[numberOfNonZeroElementsAndBias].value=instance(featId);
            ++numberOfNonZeroElementsAndBias;
          }
        }

        //add bias term
        if(bias>=0) {
          thisNode[numberOfNonZeroElementsAndBias].index = (int)numberOfFeatures+1;
          thisNode[numberOfNonZeroElementsAndBias].value = bias;
          numberOfNonZeroElementsAndBias++;
        }

        thisNode[numberOfNonZeroElementsAndBias].index = -1;
        double result = liblinear::predict(linearModel,thisNode);
        free(thisNode);
        return result;
      }

      const size_t& getNumberOfFeatures() const
      {
        return numberOfFeatures;
      }


      /**
       * Clear the model.
       */
      void clear()
      {
        liblinear::free_and_destroy_model(&linearModel);
        linearModel=nullptr;
        modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return verbose;
      }

      /**
       * Destructor.
       */
      ~LinearLogisticRegression()
      {
        liblinear::free_and_destroy_model(&linearModel);
      }

    private:
      const size_t numberOfFeatures;
      //! liblinear model defined in the liblinear library.
      liblinear::model* linearModel;
      const Penalty penaltyType;
      //! regularization parameter.
      const double C;
      const double bias;
      double tol;
      //! Indicates whether the model has been trained.
      bool modelTrained=false;
      VerboseFlag verbose = VerboseFlag::Quiet;
      const double sparseLevelCutOff;

    private:
      void constructLiblinearProblem(const MatrixXd& trainData, const VectorXd& labelData,
                                     liblinear::problem* prob,
                                     liblinear::feature_node** x_space,
                                     const double sparseLevelCutOff)
      {
        size_t numberOfData=trainData.rows();
        vector<size_t> nonZeroIndexX; nonZeroIndexX.reserve(numberOfData*numberOfFeatures);
        vector<size_t> nonZeroIndexY; nonZeroIndexY.reserve(numberOfData*numberOfFeatures);
        vector<size_t> nonZeroFeatureCountEachData(numberOfData, 0);

        prob->l = (int)numberOfData;
        size_t numberOfNonZeroElementsAndBias=0;

        for (size_t dataId=0; dataId<numberOfData; ++dataId){
          for (size_t featId=0; featId<numberOfFeatures; ++featId)
            if (fabs(trainData(dataId,featId))>sparseLevelCutOff) {
              nonZeroIndexX.push_back(dataId);
              nonZeroIndexY.push_back(featId);
              ++nonZeroFeatureCountEachData[dataId];
              ++numberOfNonZeroElementsAndBias;
            }
          ++numberOfNonZeroElementsAndBias;
        }

        prob->bias=bias;
        prob->y = (double*) malloc(prob->l*sizeof(double));
        prob->x = (liblinear::feature_node**) malloc(prob->l*sizeof(liblinear::feature_node*));
        *x_space = (liblinear::feature_node*)
          malloc((numberOfNonZeroElementsAndBias+prob->l)*sizeof(liblinear::feature_node));

        size_t currentPosInXSpace=0;
        size_t currentPosInData=0;

        for(size_t dataId=0; dataId<numberOfData; ++dataId) {
          prob->x[dataId] = &(*x_space)[currentPosInXSpace]; //the first two operator cannt collapes
          prob->y[dataId] = labelData[dataId];
          for (size_t nonZeroId=0; nonZeroId<nonZeroFeatureCountEachData[dataId]; ++nonZeroId){
            size_t xIndex=nonZeroIndexX[currentPosInData];
            size_t yIndex=nonZeroIndexY[currentPosInData];
            (*x_space)[currentPosInXSpace].index=(int)yIndex+1;
            (*x_space)[currentPosInXSpace].value=trainData(xIndex,yIndex);
            ++currentPosInXSpace;
            ++currentPosInData;
          }

          if(bias >= 0)
            (*x_space)[currentPosInXSpace++].value = bias;
          (*x_space)[currentPosInXSpace++].index = -1;
        }

        if(prob->bias >= 0) {
          prob->n=(int)numberOfFeatures+1;
          for(size_t dataId=1; dataId < (size_t)prob->l; ++dataId)
            (prob->x[dataId]-2)->index = prob->n;
          (*x_space)[currentPosInXSpace-2].index = prob->n;
        }
        else
          prob->n=(int)numberOfFeatures;
      }
    };
  }
}

#endif //MODEL_LINEAR_LOGISTIC_REGRESSION
