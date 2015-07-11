#ifndef MODEL_GRADIENT_BOOSTING_REGRESSOR
#define MODEL_GRADIENT_BOOSTING_REGRESSOR

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_BaseRegressor.hpp"
#include "../internal/_RegressionTree.hpp"
#include "../internal/_RegressionCriterion.hpp"

namespace CPPLearn
{
  namespace Models
  {
    class GradientBoostingRegressor : public _BaseRegressor<GradientBoostingRegressor> {
    public:
      using BaseRegressor = _BaseRegressor <GradientBoostingRegressor>;
      using BaseRegressor::train;
      static constexpr const char* ModelName = "GradientBoostingRegressor";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseRegressor::LossFunction;

      using Criterion = _RegressionCriterion;

      explicit GradientBoostingRegressor(const long numberOfFeatures,
                                         const long numberOfTrees,
                                         const long minSamplesInALeaf=1,
                                         const long minSamplesInANode=1,
                                         const long maxDepth=std::numeric_limits<long>::max(),
                                         const long maxNumberOfLeafNodes=-1) :
        BaseRegressor{numberOfFeatures}, _minSamplesInALeaf{minSamplesInALeaf},
        _minSamplesInANode{minSamplesInANode}, _maxDepth{maxDepth},
        _maxNumberOfLeafNodes{maxNumberOfLeafNodes},
        _numberOfFeaturesToSplit{numberOfFeatures}
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
          _trees.emplace_back(numberOfFeatures);
      }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            vector<long> trainIndices)
      {
        const VectorXd& labelData = trainLabels._labelData;

        Criterion criterion{&labelData};


        _RegressionCriterion _criterion(&labelData);

        if (_maxNumberOfLeafNodes < 0) {
          _DepthFirstBuilder<Criterion, _PresortBestSplitter> builder(_minSamplesInALeaf,
                                                                      _minSamplesInANode,
                                                                      _maxDepth,
                                                                      _numberOfFeaturesToSplit,
                                                                      &criterion);
          _buildTrees(&builder, trainData, labelData, std::move(trainIndices));
        }
        else {
          _BestFirstBuilder<Criterion, _PresortBestSplitter> builder(_minSamplesInALeaf,
                                                                     _minSamplesInANode,
                                                                     _maxDepth,
                                                                     _maxNumberOfLeafNodes,
                                                                     _numberOfFeaturesToSplit,
                                                                     &criterion);
          _buildTrees(&builder, trainData, labelData, std::move(trainIndices));
        }

        BaseRegressor::_modelTrained = true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseRegressor::_modelTrained);
        double result=0;
        for (auto& tree : _trees)
          result += tree.predictOne(instance);
        return result;
      }

      void
      _clearModel() {
        _trees.clear();
      }

      double&
      setLearningRate() {
        return _learningRate;
      }

    private:
      template<class BuilderType>
      void _buildTrees (BuilderType* builder, const MatrixXd& trainData,
                        const VectorXd& labelData, vector<long> trainIndices) {
        long numberOfData = labelData.size();
        VectorXd yPre = VectorXd(numberOfData); yPre.fill(0);
        double mean = 0;
        for (const auto& dataId : trainIndices)
          mean += trainData(dataId);
        mean /= static_cast<double>(numberOfData);

        for (const auto& dataId : trainIndices) {
          if (yPre(dataId) != 0) {
            throwException("data index repeated in GradientBoostingRegressor");
          }
          yPre(dataId) = mean;
        }

        long numberOfTrees =  _trees.size();
        for (long treeId=0; treeId<numberOfTrees; ++treeId) {
          VectorXd residual = yPre;
          try {
            builder->build(trainData, &_trees[treeId], &trainIndices);
          }
          catch (...) {
            cout<<"exception caught when training tree regressor: "<<endl;
            throw;
          }

          for (auto& dataId : trainIndices) {
            Map<const VectorXd> instance(&trainData(dataId, 0),
                                         BaseRegressor::_numberOfFeatures);
            double predictedValue=_trees[treeId].predictOne(instance);
            yPre(dataId) += _learningRate*predictedValue;
          }
        }
      }

      private:
        long _minSamplesInALeaf;
        long _minSamplesInANode;
        long _maxDepth;
        long _maxNumberOfLeafNodes;
        long _numberOfFeaturesToSplit;
        vector<_RegressionTree> _trees;
        double _learningRate = 1.0;
      };
    }
  }

#endif //MODEL_GRADIENT_BOOSTING_REGRESSOR
