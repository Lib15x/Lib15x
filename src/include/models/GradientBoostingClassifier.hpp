#ifndef MODEL_GRADIENT_BOOSTING_CLASSIFIER
#define MODEL_GRADIENT_BOOSTING_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"

namespace Lib15x
{
  namespace Models
  {
    class GradientBoostingClassifier : public _BaseClassifier<GradientBoostingClassifier> {
    public:
      using BaseClassifier = _BaseClassifier <GradientBoostingClassifier>;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "GradientBoostingClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      GradientBoostingClassifier(const long numberOfFeatures, const long numberOfClasses) :
        BaseClassifier{numberOfFeatures, numberOfClasses} { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            vector<long> trainIndices)
      {
        BaseClassifier::_modelTrained = true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        return 0;
      }

      void
      _clearModel() { }

    private:
      long somedata=0;
    };
  }
}

#endif //MODEL_GRADIENT_BOOSTING_CLASSIFIER
