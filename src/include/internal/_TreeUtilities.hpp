#ifndef _TREE_UTILITIESS
#define _TREE_UTILITIESS
#include "../core/Definitions.hpp"

namespace CPPLearn {
  struct _StackRecord {
    long _startIndex;
    long _endIndex;
    long _nodeDepth;
    bool _isLeft;
    double _impurity;
    long _numberOfConstantFeatures;
    long _parentNodeIndex;

    _StackRecord(long startIndex, long endIndex, long nodeDepth,
                 bool isLeft, double impurity, long numberOfConstantFeatures,
                 long parentNodeIndex) :
      _startIndex{startIndex}, _endIndex{endIndex}, _nodeDepth{nodeDepth},
      _isLeft{isLeft}, _impurity{impurity}, _numberOfConstantFeatures{numberOfConstantFeatures},
      _parentNodeIndex{parentNodeIndex} { }
  };

  struct SplitRecord {
    long _featureIndexToSplit;
    long _splitSampleIndex;
    double _threshold;
    double _impurityImprovement;
    double _impurityLeft;
    double _impurityRight;
    SplitRecord () : _featureIndexToSplit{-1}, _splitSampleIndex{-1},
      _threshold(-std::numeric_limits<double>::max()),
      _impurityImprovement(0.0),
      _impurityLeft(std::numeric_limits<double>::max()),
      _impurityRight(std::numeric_limits<double>::max()) { }
  };
}
#endif // _TREE_UTILITIESS
