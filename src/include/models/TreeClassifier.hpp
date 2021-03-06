#ifndef MODEL_TREE_CLASSIFIER
#define MODEL_TREE_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_ClassificationTree.hpp"
#include "../internal/_ClassificationCriterion.hpp"

namespace Lib15x
{
  namespace Models
  {
    template<double (*ImpurityRule)(const vector<long>&) = gini>
    class TreeClassifier : public _BaseClassifier<TreeClassifier<ImpurityRule> > {
    public:
      using BaseClassifier = _BaseClassifier<TreeClassifier<ImpurityRule> >;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "TreeClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      using Criterion = _ClassificationCriterion<ImpurityRule>;

      TreeClassifier(const long numberOfFeatures, const long numberOfClasses,
                     const long minSamplesInALeaf=1, const long minSamplesInANode=1,
                     const long maxDepth=std::numeric_limits<long>::max(),
                     const long maxNumberOfLeafNodes=-1) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _minSamplesInALeaf{minSamplesInALeaf}, _minSamplesInANode{minSamplesInANode},
        _maxDepth{maxDepth}, _maxNumberOfLeafNodes{maxNumberOfLeafNodes},
        _tree{numberOfFeatures, numberOfClasses}
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
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        const VectorXd& labelData=trainLabels._labelData;
        assert(weights.size()==trainLabels.size());

        vector<long> trainIndices;
        for (long dataId=0; dataId<labelData.size(); ++dataId){
          long dataRepeat=static_cast<long>(weights(dataId));
          if (static_cast<double>(dataRepeat) != weights(dataId)) {
            throwException("Error happened in TreeRegressor model: cannot handle general "
                           "sample weights=%f", weights(dataId));
          }
          for (long rep=0; rep<dataRepeat; ++rep)
            trainIndices.push_back(dataId);
        }

        Criterion criterion{&labelData, BaseClassifier::_numberOfClasses};

        if (_maxNumberOfLeafNodes < 0) {
          _DepthFirstBuilder<Criterion, _BestSplitter> builder(_minSamplesInALeaf,
                                                               _minSamplesInANode,
                                                               _maxDepth,
                                                               BaseClassifier::_numberOfFeatures,
                                                               &criterion);
          try {
            builder.build(trainData, &_tree, &trainIndices);
          }
          catch (...) {
            cout<<"exception caught when training tree classifier: "<<endl;
            throw;
          }
        }
        else {
          _BestFirstBuilder<Criterion, _BestSplitter> builder(_minSamplesInALeaf,
                                                              _minSamplesInANode,
                                                              _maxDepth,
                                                              _maxNumberOfLeafNodes,
                                                              BaseClassifier::_numberOfFeatures,
                                                              &criterion);
          try {
            builder.build(trainData, &_tree, &trainIndices);
          }
          catch (...) {
            cout<<"exception caught when training tree classifier: "<<endl;
            throw;
          }
        }

        BaseClassifier::_modelTrained = true;
      }

      double
      predictOne(const VectorXd& instance) const
      {
        assert(BaseClassifier::_modelTrained);
        vector<long> labelsCount= _tree.predictOne(instance);
        assert(static_cast<long>(labelsCount.size())==BaseClassifier::_numberOfClasses);
        auto it=std::max_element(std::begin(labelsCount), std::end(labelsCount));
        long label =(it-std::begin(labelsCount));

        return static_cast<double>(label);
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
      _ClassificationTree _tree;
    };
  }
}

#endif //MODEL_TREE_CLASSIFIER
