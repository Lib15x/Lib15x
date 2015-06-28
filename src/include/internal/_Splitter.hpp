#ifndef _SPLITTER
#define _SPLITTER
#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "./_TreeUtilities.hpp"

namespace CPPLearn {
  template<class _Criterion>
  class _Splitter {
  public:
    _Splitter(const MatrixXd* trainData, _Criterion* criterion,
              const long minSamplesInALeaf, const long numberOfFeaturesToSplit,
              vector<long>* sampleIndices) :
      _trainData{trainData},
      _criterion{criterion}, _minSamplesInALeaf{minSamplesInALeaf},
      _numberOfFeaturesToSplit{numberOfFeaturesToSplit},
      _numberOfFeatures{_trainData->cols()},
      _sampleIndices{sampleIndices}, _featureIndices(_numberOfFeatures),
      _startIndex{-1}, _endIndex{-1}
    {
      std::iota(std::begin(_featureIndices), std::end(_featureIndices),0);
    }

    void
    resetToThisNode(long startIndex, long endIndex)
    {
      _startIndex = startIndex;
      _endIndex = endIndex;
      _criterion->init(_sampleIndices, _startIndex, _endIndex);
    }

    _SplitRecord
    splitNode(const double impurity,  long* numberOfConstantFeatures)
    {
      _SplitRecord bestSplit, currentSplit;
      bestSplit._splitSampleIndex=_endIndex;
      long featIdI = _numberOfFeatures;
      long totalNumberOfConstantFeatures = *numberOfConstantFeatures;
      long numberOfSamplesInThisNode = _endIndex-_startIndex;
      long numberOfVisitedFeatures = 0;

      while (featIdI > totalNumberOfConstantFeatures &&
             numberOfVisitedFeatures < _numberOfFeaturesToSplit) {
        ++numberOfVisitedFeatures;
        long featIdJ =
          rand() % (featIdI-totalNumberOfConstantFeatures) + totalNumberOfConstantFeatures;

        currentSplit._featureIndexToSplit = _featureIndices[featIdJ];

        vector<double> dataBuffer(numberOfSamplesInThisNode);

        for (long sampleId = 0; sampleId < numberOfSamplesInThisNode; ++sampleId) {
          long dataIndex=(*_sampleIndices)[sampleId+_startIndex];
          dataBuffer[sampleId] = (*_trainData)(dataIndex, currentSplit._featureIndexToSplit);
        }

        Utilities::sortTwoArray(std::begin(dataBuffer), std::end(dataBuffer),
                                std::begin(*_sampleIndices)+_startIndex);

        if (dataBuffer[numberOfSamplesInThisNode - 1] <= dataBuffer[0] + _featureThreshold) {
          _featureIndices[featIdJ] = _featureIndices[totalNumberOfConstantFeatures];
          _featureIndices[totalNumberOfConstantFeatures] = currentSplit._featureIndexToSplit;
          ++totalNumberOfConstantFeatures;
          continue;
        }

        --featIdI;
        std::swap(_featureIndices[featIdI], _featureIndices[featIdJ]);

        _criterion->reset();
        for (long sampleId=_startIndex; sampleId<_endIndex;) {
          if (sampleId+1<_endIndex)
            while (dataBuffer[sampleId + 1-_startIndex] <=
                   dataBuffer[sampleId-_startIndex] + _featureThreshold) {
              ++sampleId;
              if (sampleId==_endIndex-1) break;
            }
          ++sampleId;
          if (sampleId >= _endIndex) break;

          currentSplit._splitSampleIndex = sampleId;

          if (((currentSplit._splitSampleIndex - _startIndex) < _minSamplesInALeaf) ||
              ((_endIndex - currentSplit._splitSampleIndex) < _minSamplesInALeaf))
            continue;

          _criterion->update(currentSplit._splitSampleIndex);
          double left = 0;
          double right = 0;

          _criterion->calculateChildrenImpurity(&left, &right);
          currentSplit._impurityImprovement = _criterion->impurityImprove(impurity);

          if (currentSplit._impurityImprovement > bestSplit._impurityImprovement) {
            _criterion->calculateChildrenImpurity(&currentSplit._impurityLeft,
                                                 &currentSplit._impurityRight);
            currentSplit._threshold = (dataBuffer[sampleId - 1-_startIndex] +
                                       dataBuffer[sampleId-_startIndex]) / 2.0;
            bestSplit = currentSplit;
          }
        }
      }

      if (bestSplit._splitSampleIndex < _endIndex) {
        long partitionEnd = _endIndex;
        long sampleId = _startIndex;
        long featId = bestSplit._featureIndexToSplit;
        while (sampleId < partitionEnd){
          long dataId = (*_sampleIndices)[sampleId];
          if ((*_trainData)(dataId, featId) <= bestSplit._threshold) {
            ++sampleId;
            continue;
          }
          --partitionEnd;
          std::swap((*_sampleIndices)[partitionEnd], (*_sampleIndices)[sampleId]);
        }
      }

      *numberOfConstantFeatures = totalNumberOfConstantFeatures;
      return bestSplit;
    }

    double
    calculateNodeImpurity() {
      return _criterion->calculateNodeImpurity();
    }

  private:
    const MatrixXd* _trainData;
    _Criterion* _criterion;
    long _minSamplesInALeaf;
    long _numberOfFeaturesToSplit;
    long _numberOfFeatures;
    vector<long>* _sampleIndices;
    vector<long> _featureIndices;
    long _startIndex;
    long _endIndex;
    double _featureThreshold=1e-7;
  };
}
#endif //_SPLITTER
