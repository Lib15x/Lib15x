#ifndef MODEL_BAGGING_CLASSIFIER
#define MODEL_BAGGING_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template <class BaseModel = TreeClassifier<> >
    class BaggingClassifier : public _BaseClassifier<BaggingClassifier<BaseModel> > {
    public:
      using BaseClassifier = _BaseClassifier<BaggingClassifier<BaseModel> >;
      using BaseClassifier::train;
      static constexpr const char* ModelName="BaggingClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      template<typename... Args>
      BaggingClassifier(const long numberOfFeatures, const long numberOfClasses,
                        const long numberOfBaseModels, const Args... args) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _numberOfBaseModels{numberOfBaseModels}
      {
        static_assert(BaseModel::ModelType == BaseClassifier::ModelType,
                      "modelType should be classification");

        _models.reserve(static_cast<unsigned>(_numberOfBaseModels));

        for (long modelCount = 0; modelCount < _numberOfBaseModels; ++modelCount)
          _models.emplace_back(numberOfFeatures, numberOfClasses, args...);
      }

      BaggingClassifier(const long numberOfFeatures, const long numberOfClasses,
                        const long numberOfBaseModels, const BaseModel& baseModel) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _numberOfBaseModels{numberOfBaseModels}, _models{_numberOfBaseModels, baseModel}
      {
        static_assert(BaseModel::ModelType == BaseClassifier::ModelType,
                      "modelType should be classification");
      }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            const vector<long>& trainIndices)
      {
        long numberOfData = trainIndices.size();
        for (long modelId=0; modelId<_numberOfBaseModels; ++modelId) {
          vector<long> sampleIndicesForThisModel;
          sampleIndicesForThisModel.reserve(numberOfData);
          for (long dataId = 0; dataId<numberOfData; ++dataId){
            long randomIndex=rand() % numberOfData;
            sampleIndicesForThisModel.push_back(trainIndices[randomIndex]);
          }
          try {
            _models[modelId].train(trainData, trainLabels, sampleIndicesForThisModel);
          }
          catch(...) {
            cout<<"Error happened when training bagging classifier, with base model Id ="
                <<modelId<<endl;
            throw;
          }
        }

        BaseClassifier::_modelTrained=true;
      }

      double predictOne(const VectorXd& instance) const
      {
        vector<long> predictedLabelsCount(BaseClassifier::_numberOfClasses, 0);
        for (long modelId = 0; modelId<_numberOfBaseModels; ++modelId){
          double predictedLabel = _models[modelId].predictOne(instance);
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
      long _numberOfBaseModels;
      vector<BaseModel> _models;
    };
  }
}

#endif //MODEL_BAGGING_CLASSIFIER
