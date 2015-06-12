#ifndef _SPLITTER
#define _SPLITTER
#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "./_TreeUtilities.hpp"
#include "./_Criterion.hpp"

namespace CPPLearn {
  class _Splitter {
  public:
    _Splitter(_Criterion criterion, long minSamplesInALeaf=1) :
      _criterion{criterion}, _minSamplesInALeaf{minSamplesInALeaf},
      _totalNumberOfSamples{0}, _numberOfFeatures{0},
      _startIndex{-1}, _endIndex{-1},
      _trainData{nullptr}, _labelData{nullptr} { }

    void
    init(const MatrixXd* trainData, const VectorXd* labelData)
    {
      _totalNumberOfSamples = trainData->rows();

      _sampleIndices = vector<long>(_totalNumberOfSamples);
      std::iota(std::begin(_sampleIndices), std::end(_sampleIndices),0);

      _numberOfFeatures = trainData->cols();
      _featureIndices = vector<long>(_numberOfFeatures);
      std::iota(std::begin(_featureIndices), std::end(_featureIndices),0);

      _labelData = labelData;
      _trainData = trainData;
    }

    void
    resetToThisNode(long startIndex, long endIndex)
    {
      _startIndex = startIndex;
      _endIndex = endIndex;
      _criterion.init(_labelData, &_sampleIndices, _startIndex, _endIndex);
    }

    _SplitRecord
    splitNode(const double impurity,  long* numberOfConstantFeatures)
    {
      _SplitRecord bestSplit, currentSplit;
      bestSplit._splitSampleIndex=_endIndex;
      long featIdI = _numberOfFeatures;
      long totalNumberOfConstantFeatures = *numberOfConstantFeatures;
      long numberOfSamplesInThisNode = _endIndex-_startIndex;

      while (featIdI > totalNumberOfConstantFeatures) {
        long featIdJ =
          rand() % (featIdI-totalNumberOfConstantFeatures) + totalNumberOfConstantFeatures;

        currentSplit._featureIndexToSplit = _featureIndices[featIdJ];

        vector<double> dataBuffer(numberOfSamplesInThisNode);

        for (long sampleId = 0; sampleId < numberOfSamplesInThisNode; ++sampleId) {
          long dataIndex=_sampleIndices[sampleId+_startIndex];
          dataBuffer[sampleId] = (*_trainData)(dataIndex, currentSplit._featureIndexToSplit);
        }

        Utilities::sortTwoArray(std::begin(dataBuffer), std::end(dataBuffer),
                                std::begin(_sampleIndices)+_startIndex);

        if (dataBuffer[numberOfSamplesInThisNode - 1] <= dataBuffer[0] + _featureThreshold) {
          _featureIndices[featIdJ] = _featureIndices[totalNumberOfConstantFeatures];
          _featureIndices[totalNumberOfConstantFeatures] = currentSplit._featureIndexToSplit;
          ++totalNumberOfConstantFeatures;
          continue;
        }

        --featIdI;
        std::swap(_featureIndices[featIdI], _featureIndices[featIdJ]);

        _criterion.reset();
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

          _criterion.update(currentSplit._splitSampleIndex);
          double left = 0;
          double right = 0;

          _criterion.calculateChildrenImpurity(&left, &right);
          currentSplit._impurityImprovement = _criterion.impurityImprove(impurity);

          if (currentSplit._impurityImprovement > bestSplit._impurityImprovement) {
            _criterion.calculateChildrenImpurity(&currentSplit._impurityLeft,
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
          long dataId = _sampleIndices[sampleId];
          if ((*_trainData)(dataId, featId) <= bestSplit._threshold) {
            ++sampleId;
            continue;
          }
          --partitionEnd;
          std::swap(_sampleIndices[partitionEnd], _sampleIndices[sampleId]);
        }
      }
      *numberOfConstantFeatures = totalNumberOfConstantFeatures;
      return bestSplit;
    }

    const vector<long>&
    nodeLabelsCount() const {
      return _criterion.nodeLabelsCount();
    }

    double
    calculateNodeImpurity() {
      return _criterion.calculateNodeImpurity();
    }

  private:
    _Criterion _criterion;
    const long _minSamplesInALeaf;
    vector<long> _sampleIndices;
    long _totalNumberOfSamples;
    vector<long> _featureIndices;
    long _numberOfFeatures;
    long _startIndex;
    long _endIndex;
    const MatrixXd* _trainData;
    const VectorXd* _labelData;
    const double _featureThreshold=1e-7;
  };
}
#endif //_SPLITTER
