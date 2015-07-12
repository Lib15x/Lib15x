#ifndef MODEL_RANDOM_FOREST_CLASSIFIER
#define MODEL_RANDOM_FOREST_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_ClassificationTree.hpp"
#include "../internal/_ClassificationCriterion.hpp"

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

      using Criterion = _ClassificationCriterion<ImpurityRule>;

      RandomForestClassifier(const long numberOfFeatures,
                             const long numberOfClasses,
                             const long numberOfTrees,
                             const long minSamplesInALeaf,
                             const long minSamplesInANode,
                             const long maxDepth,
                             const long numberOfFeaturesToSplit,
                             const long maxNumberOfLeafNodes) :
        BaseClassifier{numberOfFeatures, numberOfClasses}, _minSamplesInALeaf{minSamplesInALeaf},
        _minSamplesInANode{minSamplesInANode}, _maxDepth{maxDepth},
        _numberOfFeaturesToSplit{numberOfFeaturesToSplit},
        _maxNumberOfLeafNodes{maxNumberOfLeafNodes}
      {
        if (_minSamplesInALeaf <= 0 ||
            _minSamplesInANode <= 0 ||
            _maxDepth <= 0 ||
            _maxNumberOfLeafNodes == 0 || _maxNumberOfLeafNodes<-1 ) {
          throwException("Error happen when constructing %s, "
                         "user input regularization parameter cannot be smaller than 1: "
                         "numberOfSamplesInAleaf = : (%ld); "
                         "minSamplesInANode = : (%ld); "
                         "maxDepth = : (%ld); "
                         "maxNumberOfLeafNodes = : (%ld); ",
                         ModelName, _minSamplesInALeaf, _minSamplesInANode,
                         _maxDepth, _maxNumberOfLeafNodes);
        }

        _trees.reserve(static_cast<unsigned>(numberOfTrees));

        for (long treeCount = 0; treeCount < numberOfTrees; ++treeCount)
          _trees.emplace_back(numberOfFeatures, numberOfClasses);
      }

      RandomForestClassifier(const long numberOfFeatures,
                             const long numberOfClasses,
                             const long numberOfTrees,
                             const long minSamplesInALeaf,
                             const long minSamplesInANode,
                             const long maxDepth,
                             const long numberOfFeaturesToSplit) :
        RandomForestClassifier{numberOfFeatures, numberOfClasses, numberOfTrees,
          minSamplesInALeaf, minSamplesInANode, maxDepth, numberOfFeaturesToSplit, -1} {}

      RandomForestClassifier(const long numberOfFeatures,
                             const long numberOfClasses,
                             const long numberOfTrees) :
        RandomForestClassifier{numberOfFeatures, numberOfClasses, numberOfTrees,
          1, 1, std::numeric_limits<long>::max(),
          static_cast<long>(sqrt(static_cast<double>(numberOfFeatures))), -1} {}

      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        const VectorXd& labelData = trainLabels._labelData;
        assert(weights.size()==trainLabels.size());

        Criterion criterion{&labelData, BaseClassifier::_numberOfClasses};

        if (_maxNumberOfLeafNodes < 0) {
          _DepthFirstBuilder<Criterion, _PresortBestSplitter> builder(_minSamplesInALeaf,
                                                                      _minSamplesInANode,
                                                                      _maxDepth,
                                                                      _numberOfFeaturesToSplit,
                                                                      &criterion);
          _buildTrees(&builder, trainData, labelData, weights);
        }
        else {
          _BestFirstBuilder<Criterion, _PresortBestSplitter> builder(_minSamplesInALeaf,
                                                                     _minSamplesInANode,
                                                                     _maxDepth,
                                                                     _maxNumberOfLeafNodes,
                                                                     _numberOfFeaturesToSplit,
                                                                     &criterion);
          _buildTrees(&builder, trainData, labelData, weights);
        }

        BaseClassifier::_modelTrained = true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        vector<long> predictedLabelsCount(BaseClassifier::_numberOfClasses, 0);
        for (const auto& tree : _trees) {
          const vector<long>& labelsCount= tree.predictOne(instance);
          std::transform(std::begin(predictedLabelsCount), std::end(predictedLabelsCount),
                         std::begin(labelsCount), std::begin(predictedLabelsCount),
                         std::plus<long>());
        }

        auto maxLabelPos=std::max_element(std::begin(predictedLabelsCount),
                                          std::end(predictedLabelsCount));

        return static_cast<double>(maxLabelPos-std::begin(predictedLabelsCount));
      }

      void
      _clearModel() {
        _trees.clear();
      }

    private:
      template<class BuilderType>
      void _buildTrees (BuilderType* builder, const MatrixXd& trainData,
                        const VectorXd& labelData, const VectorXd& weights) {
        vector<long> trainIndices;
        for (long dataId=0; dataId<labelData.size(); ++dataId){
          long repeatance=static_cast<long>(weights(dataId));
          if (static_cast<double>(repeatance) != weights(dataId)) {
            throwException("Error happened in TreeRegressor model: cannot handle general "
                           "sample weights=%f", weights(dataId));
          }
          for (long rep=0; rep<repeatance; ++rep)
            trainIndices.push_back(dataId);
        }

        const long numberOfData = trainIndices.size();
        for (auto& tree : _trees) {
          vector<long> sampleIndicesForThisModel;
          sampleIndicesForThisModel.reserve(numberOfData);
          for (long dataId = 0; dataId<numberOfData; ++dataId){
            long randomIndex=rand() % numberOfData;
            sampleIndicesForThisModel.push_back(trainIndices[randomIndex]);
          }
          try {
            builder->build(trainData, &tree, &sampleIndicesForThisModel);
          }
          catch(...) {
            cout<<"Error happened when training  RandomForestClassifier, with tree Id ="
                <<&tree-&_trees[0]<<endl;
            throw;
          }
        }
      }

    private:
      long _minSamplesInALeaf;
      long _minSamplesInANode;
      long _maxDepth;
      long _numberOfFeaturesToSplit;
      long _maxNumberOfLeafNodes;
      vector<_ClassificationTree> _trees;
    };
  }
}

#endif //MODEL_RANDOM_FOREST_CLASSIFIER
