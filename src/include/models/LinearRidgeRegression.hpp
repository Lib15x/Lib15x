#ifndef MODEL_LINEAR_RIDGE_REGRESSION
#define MODEL_LINEAR_RIDGE_REGRESSION

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"

namespace CPPLearn
{
  namespace Models
  {
    /**
     * brief introduction
     */
    class LinearRidgeRegression{
    public:
      static const ProblemType ModelType = ProblemType::Regression;
      static constexpr const char* ModelName = "LinearRidgeRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::regressionSquaredNormLossFunction;
      /**
       * constructor
       */
      LinearRidgeRegression(const double regularizer, const long numberOfFeatures) :
        _regularizer{regularizer}, _numberOfFeatures{numberOfFeatures} {}

      /**
       * train the model
       *
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Regression){
          throwException("Error happen when training LinearRigdgeRegression model: "
                         "Input labelType must be Regression!\n");
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training LinearRidgeRegression model, "
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

        MatrixXd matrixC = trainData.transpose()*trainData +
          _regularizer*MatrixXd::Identity(_numberOfFeatures, _numberOfFeatures);
        _parameters=matrixC.llt().solve(trainData.transpose()*labelData);
        _modelTrained = true;
      }

      /**
       * predicting
       *
       */
      Labels predict(const MatrixXd& testData)
      {
        if (!_modelTrained){
          throwException("Error happen when predicting with LinearRidgeRegression model: "
                         "Model has not been trained yet!");
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with LinearRidgeRegression model: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, testData.cols());
        }

        long numberOfTests=testData.rows();
        Labels predictedLabels{ProblemType::Regression};
        predictedLabels._labelData.resize(numberOfTests);

        predictedLabels._labelData = testData*_parameters;
        return predictedLabels;
      }

    private:
      const double _regularizer;
      const long _numberOfFeatures;
      VectorXd _parameters;
      bool _modelTrained = false;
    };
  }
}
#endif // MODEL_LINEAR_RIDGE_REGRESSION
