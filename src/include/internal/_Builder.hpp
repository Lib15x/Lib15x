#ifndef _BUILDER
#define _BUILDER
#include "../core/Definitions.hpp"
#include "./_Splitter.hpp"
#include "./_Tree.hpp"
#include <stack>

namespace CPPLearn{

  class _DepthFirstBuilder {
  public:
    _DepthFirstBuilder(long minSamplesInALeaf, long minSamplesInANode,
                       long maxDepth, _Splitter splitter) :
      _minSamplesInANode{minSamplesInANode}, _minSamplesInALeaf{minSamplesInALeaf},
      _maxDepth{maxDepth}, _splitter{splitter} { }

    void
    build(const MatrixXd& trainData, const VectorXd& labelData, _Tree* tree)
    {
      long numberOfData = trainData.rows();
      _splitter.init(&trainData, &labelData);
      std::stack<_StackRecord> recordStack;
      long maxDepthSoFar = -1;
      recordStack.emplace(0, numberOfData, 0, false, std::numeric_limits<double>::max(),
                          0, -1);

      while (!recordStack.empty()) {
        _StackRecord stackRecord = recordStack.top();
        recordStack.pop();

        const long startIndex = stackRecord._startIndex;
        const long endIndex = stackRecord._endIndex;
        const long nodeDepth = stackRecord._nodeDepth;
        const long parentNodeIndex = stackRecord._parentNodeIndex;
        const bool isLeft = stackRecord._isLeft;
        double impurity = stackRecord._impurity;
        long numberOfConstantFeatures = stackRecord._numberOfConstantFeatures;
        const long numberOfSamplesInThisNode = endIndex - startIndex;

        _splitter.resetToThisNode(startIndex, endIndex);

        bool isLeaf = (nodeDepth >= _maxDepth) ||
          (numberOfSamplesInThisNode < _minSamplesInANode) ||
          (numberOfSamplesInThisNode < 2 * _minSamplesInALeaf);

        if (parentNodeIndex<0) impurity = _splitter.calculateNodeImpurity();

        isLeaf = isLeaf || (impurity <= _minImpurity);

        SplitRecord splitRecord;
        if (!isLeaf) {
          splitRecord = _splitter.splitNode(impurity, &numberOfConstantFeatures);
          isLeaf = isLeaf || (splitRecord._splitSampleIndex >= endIndex);
        }

        const long currentNodeIndex = tree->addNode(parentNodeIndex, isLeft,
                                                    splitRecord._featureIndexToSplit,
                                                    splitRecord._threshold);

        if (isLeaf) {
          vector<long> labelsCountOfThisNode=_splitter.nodeLabelsCount();
          tree->addLeaf(currentNodeIndex, std::move(labelsCountOfThisNode));
        }
        else {
          recordStack.emplace(splitRecord._splitSampleIndex, endIndex, nodeDepth + 1,
                              false, splitRecord._impurityRight, numberOfConstantFeatures,
                              currentNodeIndex);
          recordStack.emplace(startIndex, splitRecord._splitSampleIndex, nodeDepth + 1,
                              true, splitRecord._impurityLeft, numberOfConstantFeatures,
                              currentNodeIndex);
        }

        if (nodeDepth > maxDepthSoFar)
          maxDepthSoFar = nodeDepth;
      }
      tree->_maxDepthOfThisTree = maxDepthSoFar;
    }

  private:
    long _minSamplesInANode;
    long _minSamplesInALeaf;
    long _maxDepth;
    _Splitter _splitter;
    double _minImpurity=1e-7;
  };
}
#endif // _BUILDER
