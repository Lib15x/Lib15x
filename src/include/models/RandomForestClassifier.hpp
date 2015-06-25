#ifndef MODEL_RANDOM_FOREST_CLASSIFIER
#define MODEL_RANDOM_FOREST_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"
#include "./TreeClassifier.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template<double (*ImpurityRule)(const vector<long>&) = gini>
    class RandomForestClassifier : public _BaseClassifier<RandomForestClassifier<ImpurityRule> > {
    public:
      using BaseClassifier = _BaseClassifier<RandomForestClassifier>;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "RandomForestClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      using Tree = TreeClassifier<ImpurityRule>;

      RandomForestClassifier(const long numberOfFeatures, const long numberOfClasses,
                             const long numberOfTrees, const long numberOfFeaturesToSplit,
                             const Tree& tree) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _numberOfFeaturesToSplit{numberOfFeaturesToSplit},
        _trees{static_cast<unsigned>(numberOfTrees), tree}
      {
        for (auto& tree : _trees)
          tree.setNumberOfRandomFeauturesToSplit() = _numberOfFeaturesToSplit;
      }

      RandomForestClassifier(const long numberOfFeatures, const long numberOfClasses,
                             const long numberOfTrees, const Tree& tree) :
        RandomForestClassifier(numberOfFeatures, numberOfClasses, numberOfTrees,
                               static_cast<long>(sqrt(static_cast<double>(numberOfFeatures))),
                               tree) { }

      RandomForestClassifier(const long numberOfFeatures, const long numberOfClasses,
                             const long numberOfTrees) :
        RandomForestClassifier(numberOfFeatures, numberOfClasses, numberOfTrees,
                               static_cast<long>(sqrt(static_cast<double>(numberOfFeatures))),
                               Tree{numberOfFeatures, numberOfClasses}) { }

      template<typename... Args>
      RandomForestClassifier(const long numberOfFeatures, const long numberOfClasses,
                             const long numberOfTrees, const long numberOfFeaturesToSplit,
                             const Args... args) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _numberOfFeaturesToSplit{numberOfFeaturesToSplit}
      {
        _trees.reserve(static_cast<unsigned>(numberOfTrees));

        for (long modelCount = 0; modelCount < numberOfTrees; ++modelCount)
          _trees.emplace_back(numberOfFeatures, numberOfClasses, args...);

        for (auto& tree : _trees)
          tree.setNumberOfRandomFeauturesToSplit() = _numberOfFeaturesToSplit;
      }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            const vector<long>& trainIndices)
      {
        const long numberOfData = trainIndices.size();
        for (auto& tree : _trees) {
          vector<long> sampleIndicesForThisModel;
          sampleIndicesForThisModel.reserve(numberOfData);
          for (long dataId = 0; dataId<numberOfData; ++dataId){
            long randomIndex=rand() % numberOfData;
            sampleIndicesForThisModel.push_back(trainIndices[randomIndex]);
          }
          try {
            tree.train(trainData, trainLabels, sampleIndicesForThisModel);
          }
          catch(...) {
            cout<<"Error happened when training  RandomForestClassifier, with tree Id ="
                <<&tree-&_trees[0]<<endl;
            throw;
          }
        }
        BaseClassifier::_modelTrained=true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        vector<long> predictedLabelsCount(BaseClassifier::_numberOfClasses, 0);
        for (const auto& tree : _trees) {
          double predictedLabel = tree.predictOne(instance);
          ++predictedLabelsCount[static_cast<long>(predictedLabel)];
        }

        auto maxLabelPos=std::max_element(std::begin(predictedLabelsCount),
                                          std::end(predictedLabelsCount));

        return static_cast<double>(maxLabelPos-std::begin(predictedLabelsCount));
      }

      void
      _clearModel() {
        for (auto& tree : _trees)
          tree.clear();
      }

    private:
      long _numberOfFeaturesToSplit;
      vector<Tree> _trees;
    };
  }
}

#endif //MODEL_RANDOM_FOREST_CLASSIFIER
