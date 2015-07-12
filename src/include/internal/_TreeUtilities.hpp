#ifndef _TREE_UTILITIES
#define _TREE_UTILITIES
#include "../core/Definitions.hpp"

namespace Lib15x
{
  struct _StackRecord {
    long _startIndex;
    long _endIndex;
    long _nodeDepth;
    bool _isLeft;
    double _impurity;
    long _numberOfConstantFeatures;
    long _parentNodeIndex;

    _StackRecord(const long startIndex, const long endIndex, const long nodeDepth,
                 const bool isLeft, const double impurity, const long numberOfConstantFeatures,
                 const long parentNodeIndex) :
      _startIndex{startIndex}, _endIndex{endIndex}, _nodeDepth{nodeDepth},
      _isLeft{isLeft}, _impurity{impurity},
      _numberOfConstantFeatures{numberOfConstantFeatures},
      _parentNodeIndex{parentNodeIndex} { }
  };

  struct _PriorityQueueRecord {
    long _thisNodeIndex;
    long _startIndex;
    long _endIndex;
    long _splitSampleIndex;
    long _nodeDepth;
    bool _isLeaf;
    double _impurity;
    double _impurityLeft;
    double _impurityRight;
    double _impurityImprovement;
    _PriorityQueueRecord(const long thisNodeIndex, const long startIndex, const long endIndex,
                         const long splitSampleIndex, const long nodeDepth, const bool isLeaf,
                         const double impurity, const double impurityLeft,
                         const double impurityRight, const double impurityImprovement) :
      _thisNodeIndex{thisNodeIndex}, _startIndex{startIndex}, _endIndex{endIndex},
      _splitSampleIndex{splitSampleIndex}, _nodeDepth{nodeDepth}, _isLeaf{isLeaf},
      _impurity{impurity}, _impurityLeft{impurityLeft}, _impurityRight{impurityRight},
      _impurityImprovement{impurityImprovement} { }
  };

  struct _SplitRecord {
    long _featureIndexToSplit;
    long _splitSampleIndex;
    double _threshold;
    double _impurityImprovement;
    double _impurityLeft;
    double _impurityRight;
    _SplitRecord () : _featureIndexToSplit{-1}, _splitSampleIndex{-1},
      _threshold(-std::numeric_limits<double>::max()),
      _impurityImprovement(0.0),
      _impurityLeft(std::numeric_limits<double>::max()),
      _impurityRight(std::numeric_limits<double>::max()) { }
  };


}
#endif // _TREE_UTILITIES
