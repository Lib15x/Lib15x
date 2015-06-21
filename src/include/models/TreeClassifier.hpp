#ifndef MODEL_TREE_CLASSIFIER
#define MODEL_TREE_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_BaseClassifier.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_Tree.hpp"

namespace CPPLearn
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

      using Criterion = _Criterion<ImpurityRule>;

      TreeClassifier(const long numberOfFeatures, const long numberOfClasses,
                     const long minSamplesInALeaf=1, const long minSamplesInANode=1,
                     const long maxDepth=std::numeric_limits<long>::max(),
                     const long maxNumberOfLeafNodes=-1) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _minSamplesInALeaf{minSamplesInALeaf}, _minSamplesInANode{minSamplesInANode},
        _maxDepth{maxDepth}, _maxNumberOfLeafNodes{maxNumberOfLeafNodes}
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
        _tree._numberOfClasses = BaseClassifier::_numberOfClasses;
        _tree._numberOfFeatures = BaseClassifier::_numberOfFeatures;

        Criterion criterion{BaseClassifier::_numberOfClasses};

        std::unique_ptr<_BuilderBase> builder=nullptr;
        if (_maxNumberOfLeafNodes < 0)
          builder=std::make_unique<_DepthFirstBuilder<Criterion> >(_minSamplesInALeaf,
                                                                   _minSamplesInANode,
                                                                   _maxDepth,
                                                                   &criterion);
        else
          builder=std::make_unique<_BestFirstBuilder<Criterion> >(_minSamplesInALeaf,
                                                                  _minSamplesInANode,
                                                                  _maxDepth,
                                                                  _maxNumberOfLeafNodes,
                                                                  &criterion);

        try {
          builder->build(trainData, labelData, &_tree, &trainIndices);
        }
        catch (...) {
          cout<<"exception caught when training tree classifier: "<<endl;
          throw;
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
      _Tree _tree;
    };
  }
}

#endif //MODEL_TREE_CLASSIFIER
