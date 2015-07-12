#ifndef _CLASSIFICATION_TREE
#define _CLASSIFICATION_TREE
#include "./_BaseTree.hpp"

namespace Lib15x
{
  class _ClassificationTree : public _BaseTree<_ClassificationTree> {
  public:
    using BaseTree = _BaseTree<_ClassificationTree>;

    _ClassificationTree (const long numberOfFeatures, const long numberOfClasses) :
      BaseTree{numberOfFeatures}, _numberOfClasses{numberOfClasses} { }

    void _reset()
    {
      _leafNodeToLabel.clear();
    }

    void
    addLeaf(const long nodeIndex, vector<long> labelsCount)
    {
      assert(static_cast<long>(labelsCount.size())==_numberOfClasses);
      _leafNodeToLabel.push_back(std::make_pair(nodeIndex, std::move(labelsCount)));
      _nodes[nodeIndex]._isLeaf = true;
    }

    const vector<long>&
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
    vector<std::pair<long, vector<long> > > _leafNodeToLabel;
    long _numberOfClasses;
  };
}

#endif // _CLASSIFICATION_TREE
