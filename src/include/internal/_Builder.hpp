#ifndef _BUILDER
#define _BUILDER
#include "../core/Definitions.hpp"
#include "./_Splitter.hpp"
#include "./_Tree.hpp"
#include <stack>
#include <queue>

namespace CPPLearn{

  template<class _Criterion>
  class _DepthFirstBuilder {
  public:
    using Splitter = _Splitter<_Criterion>;

    _DepthFirstBuilder(long minSamplesInALeaf, long minSamplesInANode,
                       long maxDepthAllowed, _Criterion* criterion) :
      _minSamplesInANode{minSamplesInANode}, _minSamplesInALeaf{minSamplesInALeaf},
      _maxDepthAllowed{maxDepthAllowed}, _criterion{criterion} { }

    void
    build(const MatrixXd& trainData, const VectorXd& labelData,
          _Tree* tree, vector<long>* sampleIndices)
    {
      const long numberOfData = trainData.rows();
      Splitter splitter(&trainData, &labelData, _criterion,
                        _minSamplesInALeaf, sampleIndices);
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

        splitter.resetToThisNode(startIndex, endIndex);

        bool isLeaf = (nodeDepth >= _maxDepthAllowed) ||
          (numberOfSamplesInThisNode < _minSamplesInANode) ||
          (numberOfSamplesInThisNode < 2 * _minSamplesInALeaf);

        if (parentNodeIndex<0) impurity = splitter.calculateNodeImpurity();

        isLeaf = isLeaf || (impurity <= _minImpurity);

        _SplitRecord splitRecord;
        if (!isLeaf) {
          splitRecord = splitter.splitNode(impurity, &numberOfConstantFeatures);
          isLeaf = isLeaf || (splitRecord._splitSampleIndex >= endIndex);
        }

        const long currentNodeIndex = tree->addNode(parentNodeIndex, isLeft,
                                                    splitRecord._featureIndexToSplit,
                                                    splitRecord._threshold);

        if (isLeaf) {
          vector<long> labelsCountOfThisNode = splitter.nodeLabelsCount();
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
    long _maxDepthAllowed;
    _Criterion* _criterion;
    const double _minImpurity=1e-7;
  };

  template<class _Criterion>
  class _BestFirstBuilder {
  public:
    using Splitter= _Splitter<_Criterion>;

    _BestFirstBuilder(const long minSamplesInALeaf, const long minSamplesInANode,
                      const long maxDepthAllowed, const long maxNumberOfLeafNodes,
                      _Criterion* criterion) :
      _minSamplesInANode{minSamplesInANode}, _minSamplesInALeaf{minSamplesInALeaf},
      _maxDepthAllowed{maxDepthAllowed}, _maxNumberOfLeafNodes{maxNumberOfLeafNodes},
      _criterion{criterion} { }

    void
    build(const MatrixXd& trainData, const VectorXd& labelData, _Tree* tree,
             vector<long>* sampleIndices)
    {
      auto compareRecord =
        [](const _PriorityQueueRecord& a, const _PriorityQueueRecord& b)
        {return a._impurityImprovement < b._impurityImprovement;};

      using PriorityQueue =
        std::priority_queue<_PriorityQueueRecord, vector<_PriorityQueueRecord>, decltype(compareRecord)>;

      const long numberOfData = trainData.rows();

      Splitter splitter(&trainData, &labelData, _criterion,
                        _minSamplesInALeaf, sampleIndices);
      splitter.resetToThisNode(0, numberOfData);

      PriorityQueue recordQueue(compareRecord);

      long numberOfInnerNodes = _maxNumberOfLeafNodes - 1;
      long maxDepthSoFar = -1;

      double impurity = splitter.calculateNodeImpurity();

      _PriorityQueueRecord record =
        _splitAndAddNode(tree, &splitter,  0, numberOfData, impurity, true, -1, 0);

      recordQueue.emplace(record);

      while (!recordQueue.empty()) {
        _PriorityQueueRecord record = recordQueue.top();
        recordQueue.pop();

        bool isLeaf = (record._isLeaf || numberOfInnerNodes <= 0);
        if (record._nodeDepth > maxDepthSoFar)
          maxDepthSoFar = record._nodeDepth;

        if (isLeaf) continue;

        --numberOfInnerNodes;

        _PriorityQueueRecord leftRecord =
          _splitAndAddNode(tree, &splitter, record._startIndex, record._splitSampleIndex,
                           record._impurityLeft, true, record._thisNodeIndex,
                           record._nodeDepth + 1);
        _PriorityQueueRecord rightRecord =
          _splitAndAddNode(tree, &splitter, record._splitSampleIndex, record._endIndex,
                           record._impurityRight, false, record._thisNodeIndex,
                           record._nodeDepth + 1);

        recordQueue.emplace(leftRecord);
        recordQueue.emplace(rightRecord);
      }

      tree->_maxDepthOfThisTree = maxDepthSoFar;
    }

    _PriorityQueueRecord
    _splitAndAddNode(_Tree* tree, Splitter* splitter,
                     const long startIndex, const long endIndex,
                     const double impurity, const bool isLeft, const long parentNodeIndex,
                     const long nodeDepth)
    {
      long numberOfConstantFeatures = 0;

      splitter->resetToThisNode(startIndex, endIndex);

      long numberOfSamplesInThisNode = endIndex - startIndex;
      bool isLeaf = (nodeDepth > _maxDepthAllowed) ||
        (numberOfSamplesInThisNode < _minSamplesInANode) ||
        (numberOfSamplesInThisNode < 2 * _minSamplesInALeaf) ||
        (impurity <= _minImpurity);

      _SplitRecord splitRecord;

      if (!isLeaf) {
        splitRecord = splitter->splitNode(impurity, &numberOfConstantFeatures);
        isLeaf = isLeaf || (splitRecord._splitSampleIndex >= endIndex);
      }

      const long currentNodeIndex = tree->addNode(parentNodeIndex, isLeft,
                                                  splitRecord._featureIndexToSplit,
                                                  splitRecord._threshold);

      if (isLeaf) {
        vector<long> labelsCountOfThisNode=splitter->nodeLabelsCount();
        tree->addLeaf(currentNodeIndex, std::move(labelsCountOfThisNode));
      }

      _PriorityQueueRecord record(currentNodeIndex, startIndex, endIndex,
                                  splitRecord._splitSampleIndex, nodeDepth, isLeaf,
                                  impurity, splitRecord._impurityLeft,
                                  splitRecord._impurityRight,
                                  splitRecord._impurityImprovement);

      return record;
    }

  private:
    long _minSamplesInANode;
    long _minSamplesInALeaf;
    long _maxDepthAllowed;
    long _maxNumberOfLeafNodes;
    _Criterion* _criterion;
    double _minImpurity=1e-7;
  };
}
#endif // _BUILDER
