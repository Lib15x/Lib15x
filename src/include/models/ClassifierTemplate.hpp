#ifndef MODEL_CLASSIFIER
#define MODEL_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"

namespace CPPLearn
{
  namespace Models
  {
    class ClassifierTemplate : public _BaseClassifier<ClassifierTemplate> {
    public:
      using BaseClassifier = _BaseClassifier <ClassifierTemplate >;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "ClassifierTemplate";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      ClassifierTemplate(const long numberOfFeatures, const long numberOfClasses) :
        BaseClassifier{numberOfFeatures, numberOfClasses} { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        assert(weights.size()==trainLabels.size());
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

#endif //MODEL_CLASSIFIER
