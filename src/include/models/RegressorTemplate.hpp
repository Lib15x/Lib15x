#ifndef MODEL_REGRESSOR
#define MODEL_REGRESSOR

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
    class RegressorTemplate : public _BaseRegressor<RegressorTemplate> {
    public:
      using BaseRegressor = _BaseRegressor<RegressorTemplate>;
      using BaseRegressor::train;
      static constexpr const char* ModelName = "RegressorTemplate";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseRegressor::LossFunction;
      /**
       * constructor
       */
      explicit RegressorTemplate(const long numberOfFeatures) :
        BaseRegressor{numberOfFeatures} { }

      /**
       * train the model
       *
       */
      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            const vector<long>& trainIndices)
      {
        BaseRegressor::_modelTrained = true;
      }

      /**
       * predicting
       *
       */
      double predictOne(const VectorXd& instance) const
      {
        assert(BaseRegressor::_modelTrained);
        return 0;
      }

      void _clearModel() { }

    private:
      long somedata=0;
    };
  }
}
#endif // MODEL_REGRESSOR
