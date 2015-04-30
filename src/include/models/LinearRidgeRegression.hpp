#ifndef MODEL_LINEAR_RIDGE_REGRESSION
#define MODEL_LINEAR_RIDGE_REGRESSION

#include <core/Definitions.hpp>
#include <core/Utilities.hpp>

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
      static constexpr const char* ModelName="LinearRidgeRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::regressionSquaredNormLossFunction;

      /**
       * constructor
       */
      LinearRidgeRegression(const double regularizer_, const size_t numberOfFeatures_) :
        regularizer{regularizer_}, numberOfFeatures{numberOfFeatures_} {}

      /**
       * train the model
       *
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels.labelType != ProblemType::Regression){
          throwException("Error happen when training LinearRigdgeRegression model: "
                         "Input labelType must be Regression!\n");
        }

        const VectorXd& labelData=trainLabels.labelData;

        if ((unsigned)trainData.cols() != numberOfFeatures){
          throwException("Error happen when training LinearRidgeRegression model, "
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

        MatrixXd matrixC = trainData.transpose()*trainData +
          regularizer*MatrixXd::Identity(numberOfFeatures, numberOfFeatures);
        parameters=matrixC.llt().solve(trainData.transpose()*labelData);
        modelTrained = true;
      }

      /**
       * predicting
       *
       */
      Labels predict(const MatrixXd& testData)
      {
        if (!modelTrained){
          throwException("Error happen when predicting with LinearRidgeRegression model: "
                         "Model has not been trained yet!");
        }

        if ((unsigned)testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting with LinearRidgeRegression model: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, testData.cols());
        }

        size_t numberOfTests=testData.rows();
        Labels predictedLabels(ProblemType::Regression);
        predictedLabels.labelData.resize(numberOfTests);

        predictedLabels.labelData = testData*parameters;
        return predictedLabels;
      }

    private:
      const double regularizer;
      const size_t numberOfFeatures;
      VectorXd parameters;
      bool modelTrained = false;
    };
  }
}
#endif // MODEL_LINEAR_RIDGE_REGRESSION
