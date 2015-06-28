#ifndef _REGRESSION_TREE
#define _REGRESSION_TREE
#include "./_BaseTree.hpp"

namespace CPPLearn {

  class _RegressionTree : public _BaseTree<_RegressionTree> {
  public:
    using BaseTree = _BaseTree<_RegressionTree>;

    explicit _RegressionTree (const long numberOfFeatures) : BaseTree{numberOfFeatures}{ }

    void _reset()
    {
      _leafNodeToLabel.clear();
    }

    void
    addLeaf(const long nodeIndex, const double label)
    {
      _leafNodeToLabel.push_back(std::make_pair(nodeIndex, label));
      _nodes[nodeIndex]._isLeaf = true;
    }

    double
    predictOne(const VectorXd& instance) const
    {
      assert(instance.size()==_numberOfFeatures);
      long leafIndex=_headNodeIndex;
      while (!_nodes[leafIndex]._isLeaf){
        long splitFeatIndex=_nodes[leafIndex]._featureIndex;
        leafIndex = instance(splitFeatIndex) < _nodes[leafIndex]._threshold ?
          _nodes[leafIndex]._leftChildIndex : _nodes[leafIndex]._rightChildIndex;
      }

      auto it = std::find_if(std::begin(_leafNodeToLabel), std::end(_leafNodeToLabel),
                             [&leafIndex](auto i){return i.first==leafIndex;});
      if (it==std::end(_leafNodeToLabel)){
        throwException("Error happened in internal tree predicting method: "
                       "cannot find the predicted data for leafIndex %ld\n.",
                       leafIndex);
      }
      return it->second;
    }

  public:
    vector<std::pair<long, double> > _leafNodeToLabel;
  };
}

#endif // _REGRESSION_TREE
