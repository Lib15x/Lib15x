#ifndef _CRITERION
#define _CRITERION

#include "../core/Definitions.hpp"

namespace CPPLearn {

  class _Criterion {
  public:
    explicit _Criterion(const long numberOfClasses) : _labelData{nullptr}, _sampleIndices{nullptr},
      _numberOfSamplesInThisNode{0}, _startIndex{-1}, _endIndex{-1}, _currentPosition{-1},
      _numberOfClasses{numberOfClasses}, _labelsCountTotal(numberOfClasses,0),
      _labelsCountLeft(numberOfClasses,0), _labelsCountRight(numberOfClasses,0),
      _numberOfSamplesOnLeft{0}, _numberOfSamplesOnRight{0} { }

    void
    init(const VectorXd* labelData, const vector<long>* sampleIndices,
         const long startIndex, const long endIndex)
    {
      _labelData = labelData;
      _sampleIndices = sampleIndices;
      _startIndex = startIndex;
      _endIndex = endIndex;
      _numberOfSamplesInThisNode = endIndex - startIndex;
      std::fill(std::begin(_labelsCountTotal), std::end(_labelsCountTotal), 0);

      for (long sampleId = _startIndex; sampleId < _endIndex; ++sampleId){
        long dataId = _sampleIndices->at(sampleId);
        long thisLabel =  static_cast<long>((*_labelData)(dataId));
        ++_labelsCountTotal[thisLabel];
      }
      reset();
    }

    void
    reset()
    {
      _numberOfSamplesOnLeft=0;
      _numberOfSamplesOnRight=_numberOfSamplesInThisNode;
      _currentPosition = _startIndex;

      std::copy(std::begin(_labelsCountTotal), std::end(_labelsCountTotal),
                std::begin(_labelsCountRight));

      std::fill(std::begin(_labelsCountLeft), std::end(_labelsCountLeft), 0);
    }

    void
    update(const long newPos)
    {
      for (long pos = _currentPosition; pos < newPos; ++pos){
        long dataId = _sampleIndices->at(pos);
        long thisLabel = static_cast<long>((*_labelData)(dataId));
        ++_labelsCountLeft[thisLabel];
        --_labelsCountRight[thisLabel];
      }
      _numberOfSamplesOnLeft += newPos-_currentPosition;
      _numberOfSamplesOnRight -= newPos-_currentPosition;
      _currentPosition = newPos;
    }

    double
    calculateNodeImpurity() const {
      double entropy = 1.0;
      for (long classId=0; classId<_numberOfClasses; ++classId) {
        double fraction = static_cast<double>(_labelsCountTotal[classId])/
          static_cast<double>(_numberOfSamplesInThisNode);
        entropy *= fraction;
      }
      return entropy*static_cast<double> (_numberOfSamplesInThisNode);
    }

    void
    calculateChildrenImpurity(double* impurityLeft,
                              double* impurityRight) const
    {
      *impurityLeft=1.0;
      *impurityRight=1.0;
      for (long classId=0; classId < _numberOfClasses; ++classId) {
        *impurityLeft *= static_cast<double>(_labelsCountLeft[classId]);
        *impurityRight *= static_cast<double>(_labelsCountRight[classId]);
      }
      *impurityLeft/=static_cast<double>(_numberOfSamplesOnLeft);
      *impurityRight/=static_cast<double>(_numberOfSamplesOnRight);
    }

    const vector<long>&
    nodeLabelsCount() const
    {
      return _labelsCountTotal;
    }

    double
    impurityImprove(const double impurity)
    {
      double impurityLeft = 0;
      double impurityRight = 0;
      long numberOfData=_labelData->size();
      calculateChildrenImpurity(&impurityLeft, &impurityRight);
      double weightTotal = static_cast<double>(_numberOfSamplesInThisNode)/
        static_cast<double>(numberOfData);
      double weightLeft = static_cast<double>(_numberOfSamplesOnLeft)/
        static_cast<double>(_numberOfSamplesInThisNode);
      double weightRight = static_cast<double>(_numberOfSamplesOnRight)/
        static_cast<double>(_numberOfSamplesInThisNode);

      double impurityImprove =
        weightTotal * (impurity - weightRight * impurityRight - weightLeft * impurityLeft);

      return impurityImprove;
    }

  private:
    const VectorXd* _labelData;
    const vector<long>* _sampleIndices;
    long _numberOfSamplesInThisNode;
    long _startIndex;
    long _endIndex;
    long _currentPosition;
    const long _numberOfClasses;
    vector<long> _labelsCountTotal;
    vector<long> _labelsCountLeft;
    vector<long> _labelsCountRight;
    long _numberOfSamplesOnLeft;
    long _numberOfSamplesOnRight;
  };
}
#endif //_CRITERION
