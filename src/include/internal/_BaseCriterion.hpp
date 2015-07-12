#ifndef _BASE_CRITERION
#define _BASE_CRITERION

#include "../core/Definitions.hpp"

namespace Lib15x
{
  template <class DerivedCriterion>
  class _BaseCriterion {
  public:
    _BaseCriterion(const VectorXd* labelData) : _labelData{labelData}, _sampleIndices{nullptr},
      _numberOfSamplesInThisNode{0}, _startIndex{-1}, _endIndex{-1}, _currentPosition{-1},
      _numberOfSamplesOnLeft{0}, _numberOfSamplesOnRight{0} { }

    void
    init(const vector<long>* sampleIndices, const long startIndex, const long endIndex)
    {
      _sampleIndices = sampleIndices;
      _startIndex = startIndex;
      _endIndex = endIndex;
      _numberOfSamplesInThisNode = endIndex - startIndex;

      static_cast<DerivedCriterion*>(this)->_init();
      reset();
    }

    void
    reset()
    {
      _numberOfSamplesOnLeft=0;
      _numberOfSamplesOnRight=_numberOfSamplesInThisNode;
      _currentPosition = _startIndex;
      static_cast<DerivedCriterion*>(this)->_reset();
    }

    void
    update(const long newPos)
    {
      _numberOfSamplesOnLeft += newPos-_currentPosition;
      _numberOfSamplesOnRight -= newPos-_currentPosition;
      static_cast<DerivedCriterion*>(this)->_update(newPos);
      _currentPosition = newPos;
    }

    double
    impurityImprove(const double impurity)
    {
      double impurityLeft = 0;
      double impurityRight = 0;
      long numberOfData=_labelData->size();
      static_cast<DerivedCriterion*>(this)->
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

  protected:
    const VectorXd* _labelData;
    const vector<long>* _sampleIndices;
    long _numberOfSamplesInThisNode;
    long _startIndex;
    long _endIndex;
    long _currentPosition;
    long _numberOfSamplesOnLeft;
    long _numberOfSamplesOnRight;
  };
}
#endif //_BASE_CRITERION
