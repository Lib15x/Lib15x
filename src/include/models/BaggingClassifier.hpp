#ifndef MODEL_BAGGING_CLASSIFIER
#define MODEL_BAGGING_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "./TreeClassifier.hpp"
#include "../internal/_BaseClassifier.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template <class BaseModel = TreeClassifier<> >
    class BaggingClassifier : public _BaseClassifier<BaggingClassifier<BaseModel> > {
    public:
      using BaseClassifier = _BaseClassifier<BaggingClassifier>;
      using BaseClassifier::train;
      static constexpr const char* ModelName="BaggingClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      template<typename... Args>
      BaggingClassifier(const long numberOfFeatures, const long numberOfClasses,
                        const long numberOfBaseModels, const Args... args) :
        BaseClassifier{numberOfFeatures, numberOfClasses}
      {
        static_assert(BaseModel::ModelType == BaseClassifier::ModelType,
                      "modelType should be classification");

        _models.reserve(static_cast<unsigned>(numberOfBaseModels));

        for (long modelCount = 0; modelCount < numberOfBaseModels; ++modelCount)
          _models.emplace_back(numberOfFeatures, numberOfClasses, args...);
      }

      BaggingClassifier(const long numberOfFeatures, const long numberOfClasses,
                        const long numberOfBaseModels, const BaseModel& baseModel) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _models{numberOfBaseModels, baseModel}
      {
        static_assert(BaseModel::ModelType == BaseClassifier::ModelType,
                      "modelType should be classification");
      }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        assert(weights.size()==trainLabels.size());
        long numberOfData = trainLabels.size();
        for (auto& model : _models) {
          VectorXd weightsForThisModel(numberOfData); weightsForThisModel.fill(0);
          for (long dataId = 0; dataId<numberOfData; ++dataId){
            long randomIndex=rand() % numberOfData;
            weightsForThisModel(randomIndex) += weights(randomIndex);
          }
          try {
            model.train(trainData, trainLabels, weightsForThisModel);
          }
          catch(...) {
            cout<<"Error happened when training bagging classifier, with base model Id ="
                <<&model-&_models[0]<<endl;
            throw;
          }
        }

        BaseClassifier::_modelTrained=true;
      }

      double predictOne(const VectorXd& instance) const
      {
        vector<long> predictedLabelsCount(BaseClassifier::_numberOfClasses, 0);
        for (const auto& model : _models) {
          double predictedLabel = model.predictOne(instance);
          ++predictedLabelsCount[static_cast<long>(predictedLabel)];
        }

        auto maxLabelPos=std::max_element(std::begin(predictedLabelsCount),
                                          std::end(predictedLabelsCount));

        return static_cast<double>(maxLabelPos-std::begin(predictedLabelsCount));
      }

      void _clearModel()
      {
        for (auto& model : _models)
          model.clear();
      }

    private:
      vector<BaseModel> _models;
    };
  }
}

#endif //MODEL_BAGGING_CLASSIFIER
