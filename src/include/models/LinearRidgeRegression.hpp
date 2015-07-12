#ifndef MODEL_LINEAR_RIDGE_REGRESSION
#define MODEL_LINEAR_RIDGE_REGRESSION

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseRegressor.hpp"

namespace Lib15x
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
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        assert(weights.size()==trainData.rows());
        const long numberOfData=trainData.rows();
        const long numberOfFeatures = BaseRegressor::_numberOfFeatures;
        const VectorXd& labelData=trainLabels._labelData;

        long index=0;
        for (; index<numberOfData; ++index)
          if(weights[index] != 1.0) break;

        if (index == numberOfData) {
          MatrixXd matrixC = trainData.transpose()*trainData +
            _regularizer*MatrixXd::Identity(_numberOfFeatures, _numberOfFeatures);

          _parameters=matrixC.llt().solve(trainData.transpose()*labelData);
          BaseRegressor::_modelTrained = true;
          return;
        }

        MatrixXd weightedData(numberOfData, numberOfFeatures);
        VectorXd weightedLabelData(numberOfData);

        for (long dataId=0; dataId<numberOfData; ++dataId){
          double sqtWeight=sqrt(weights(dataId));
          weightedData.row(dataId) = sqtWeight*trainData.row(dataId);
          weightedLabelData(dataId) = sqtWeight*labelData(dataId);
        }

        MatrixXd matrixC = weightedData.transpose()*weightedData +
          _regularizer*MatrixXd::Identity(_numberOfFeatures, _numberOfFeatures);

        _parameters=matrixC.llt().solve(trainData.transpose()*weightedLabelData);

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
