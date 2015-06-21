#ifndef MODEL_LINEAR_RIDGE_REGRESSION
#define MODEL_LINEAR_RIDGE_REGRESSION

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseRegressor.hpp"

namespace CPPLearn
{
  namespace Models
  {
    /**
     * brief introduction
     */
    class LinearRidgeRegression : public _BaseRegressor<LinearRidgeRegression> {
    public:
      using BaseRegressor = _BaseRegressor<LinearRidgeRegression >;
      using BaseRegressor::train;
      static constexpr const char* ModelName = "LinearRidgeRegression";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseRegressor::LossFunction;
      /**
       * constructor
       */
      LinearRidgeRegression(const long numberOfFeatures, const double regularizer) :
        BaseRegressor{numberOfFeatures}, _regularizer{regularizer} { }

      /**
       * train the model
       *
       */
      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            const vector<long>& trainIndices)
      {
        const long numberOfFeatures = BaseRegressor::_numberOfFeatures;
        const long numberOfData=trainIndices.size();
        const VectorXd& labelData=trainLabels._labelData;
        MatrixXd matrixC(numberOfFeatures, numberOfFeatures);

        long index=0;
        if (trainData.rows() == numberOfData)
          for (; index<numberOfData; ++index)
            if(trainIndices[index] != index) break;

        if (index != trainData.rows()) {
          MatrixXd thisData(numberOfData, numberOfFeatures);
          for (long dataId=0; dataId<numberOfData; ++dataId)
            thisData.row(dataId)=trainData.row(trainIndices[dataId]);
          matrixC = thisData.transpose()*thisData +
            _regularizer*MatrixXd::Identity(_numberOfFeatures, _numberOfFeatures);
        }
        else
          matrixC = trainData.transpose()*trainData +
            _regularizer*MatrixXd::Identity(_numberOfFeatures, _numberOfFeatures);

        _parameters=matrixC.llt().solve(trainData.transpose()*labelData);

        BaseRegressor::_modelTrained = true;
      }

      /**
       * predicting
       *
       */
      double predictOne(const VectorXd& instance) const
      {
        assert(BaseRegressor::_modelTrained);
        return _parameters.dot(instance);
      }

      void _clearModel() { }

    private:
      double _regularizer;
      VectorXd _parameters;
    };
  }
}
#endif // MODEL_LINEAR_RIDGE_REGRESSION
