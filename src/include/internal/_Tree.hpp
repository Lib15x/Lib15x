#ifndef _TREE
#define _TREE
#include "../core/Definitions.hpp"

namespace CPPLearn {

  class _Tree {
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
    _Tree () :
      _maxDepthOfThisTree{0}, _headNodeIndex{-1} , _nodeCount{0},
      _numberOfClasses{0}, _numberOfFeatures{0} { }

    void reset()
    {
      _maxDepthOfThisTree = 0;
      _headNodeIndex = -1;
      _nodes.clear();
      _leafNodeToLabel.clear();
    }

  public:
    long _maxDepthOfThisTree;
    long _headNodeIndex;
    vector<_Node> _nodes;
    long _nodeCount;
    vector<std::pair<long, vector<long> > > _leafNodeToLabel;
    long _numberOfClasses;
    long _numberOfFeatures;

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
  };
}

#endif // _TREE
