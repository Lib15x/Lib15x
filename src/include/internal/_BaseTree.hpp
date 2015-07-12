#ifndef _BASE_TREE
#define _BASE_TREE
#include "../core/Definitions.hpp"

namespace Lib15x
{
  template<class DerivedTree>
  class _BaseTree {
  public:
    struct _Node {
      long _leftChildIndex;
      long _rightChildIndex;
      long _featureIndex;
      double _threshold;
      bool _isLeaf;

      _Node () : _leftChildIndex{-1}, _rightChildIndex{-1},
        _featureIndex{-1}, _threshold{-std::numeric_limits<double>::max()}, _isLeaf{false} { }

      _Node (const long featureIndex, const double threshold) :
        _leftChildIndex{-1}, _rightChildIndex{-1},
        _featureIndex{featureIndex}, _threshold{threshold}, _isLeaf{false} { }
    };

  public:
    explicit _BaseTree (const long numberOfFeatures) :
      _maxDepthOfThisTree{0}, _headNodeIndex{-1} , _nodeCount{0},
      _numberOfFeatures{numberOfFeatures} { }

    void reset()
    {
      _maxDepthOfThisTree = 0;
      _headNodeIndex = -1;
      _nodes.clear();
      static_cast<DerivedTree*>(this)->_reset();
    }

    long
    addNode(long parentNodeIndex, const bool isLeft, const long featureIndex,
            const double threshold)
    {
      _nodes.emplace_back(featureIndex, threshold);
      ++_nodeCount;

      if (parentNodeIndex < 0) {
        _headNodeIndex = _nodeCount-1;
        return _headNodeIndex;
      }

      isLeft ? _nodes[parentNodeIndex]._leftChildIndex = _nodeCount-1 :
        _nodes[parentNodeIndex]._rightChildIndex = _nodeCount-1;
      return _nodeCount-1;
    }

  public:
    long _maxDepthOfThisTree;
    long _headNodeIndex;
    vector<_Node> _nodes;
    long _nodeCount;
    long _numberOfFeatures;
  };
}

#endif // _BASE_TREE
