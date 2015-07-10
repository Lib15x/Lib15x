#ifndef MODEL_TREE_REGRESSOR
#define MODEL_TREE_REGRESSOR

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseRegressor.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_RegressionTree.hpp"
#include "../internal/_RegressionCriterion.hpp"

namespace CPPLearn
{
  namespace Models
  {
    class TreeRegressor : public _BaseRegressor<TreeRegressor> {
    public:
      using BaseRegressor = _BaseRegressor<TreeRegressor>;
      using BaseRegressor::train;
      static constexpr const char* ModelName = "TreeRegressor";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseRegressor::LossFunction;

      using Criterion = _RegressionCriterion;

      explicit TreeRegressor(const long numberOfFeatures,
                             const long minSamplesInALeaf=1, const long minSamplesInANode=1,
                             const long maxDepth=std::numeric_limits<long>::max(),
                             const long maxNumberOfLeafNodes=-1) :
        BaseRegressor{numberOfFeatures}, _minSamplesInALeaf{minSamplesInALeaf},
        _minSamplesInANode{minSamplesInANode}, _maxDepth{maxDepth},
        _maxNumberOfLeafNodes{maxNumberOfLeafNodes},
        _numberOfFeaturesToSplit{numberOfFeatures},
        _tree{numberOfFeatures}
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
      }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels,
            vector<long> trainIndices)
      {
        const VectorXd& labelData=trainLabels._labelData;

        Criterion criterion{&labelData};

        if (_maxNumberOfLeafNodes < 0) {
          _DepthFirstBuilder<Criterion, _BestSplitter> builder(_minSamplesInALeaf,
                                                               _minSamplesInANode,
                                                               _maxDepth,
                                                               _numberOfFeaturesToSplit,
                                                               &criterion);
          try {
            builder.build(trainData, &_tree, &trainIndices);
          }
          catch (...) {
            cout<<"exception caught when training tree regressor: "<<endl;
            throw;
          }
        }
        else {
          _BestFirstBuilder<Criterion, _BestSplitter> builder(_minSamplesInALeaf,
                                                              _minSamplesInANode,
                                                              _maxDepth,
                                                              _maxNumberOfLeafNodes,
                                                              _numberOfFeaturesToSplit,
                                                              &criterion);
          try {
            builder.build(trainData, &_tree, &trainIndices);
          }
          catch (...) {
            cout<<"exception caught when training tree regressor: "<<endl;
            throw;
          }
        }

        BaseRegressor::_modelTrained = true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseRegressor::_modelTrained);
        double label = _tree.predictOne(instance);
        return label;
      }

      long&
      setNumberOfRandomFeauturesToSplit() {
        return _numberOfFeaturesToSplit;
      }

      void
      _clearModel()
      {
        _tree.reset();
      }

    private:
      long _minSamplesInALeaf;
      long _minSamplesInANode;
      long _maxDepth;
      long _maxNumberOfLeafNodes;
      long _numberOfFeaturesToSplit;
      _RegressionTree _tree;
    };
  }
}

#endif //MODEL_TREE_REGRESSOR
