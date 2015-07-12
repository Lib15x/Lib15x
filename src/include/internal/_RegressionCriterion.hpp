#ifndef _REGRESSION_CRITERION
#define _REGRESSION_CRITERION

#include "../core/Definitions.hpp"
#include "./_BaseCriterion.hpp"

namespace Lib15x
{
  class _RegressionCriterion :
    public _BaseCriterion<_RegressionCriterion> {
  public:
    using BaseCriterion = _BaseCriterion<_RegressionCriterion>;

    explicit _RegressionCriterion(const VectorXd* labelData) :
      BaseCriterion{labelData}, _meanLeft{0.0}, _meanRight{0.0}, _meanTotal{0.0},
      _sqSumLeft{0.0}, _sqSumRight{0.0}, _sqSumTotal{0.0}, _varLeft{0.0}, _varRight{0.0},
      _sumLeft{0.0}, _sumRight{0.0}, _sumTotal{0.0} { }

    void
    _init()
    {
      long startIndex=BaseCriterion::_startIndex;
      long endIndex=BaseCriterion::_endIndex;

      for (long sampleId = startIndex; sampleId < endIndex; ++sampleId) {
        long dataId = BaseCriterion::_sampleIndices->at(sampleId);
        double thisLabel =  (*BaseCriterion::_labelData)(dataId);
        _sumTotal+=thisLabel;
        _sqSumTotal+=thisLabel*thisLabel;
      }

      _meanTotal=_sumTotal/static_cast<double>(BaseCriterion::_numberOfSamplesInThisNode);
    }

    void
    _reset()
    {
      _meanRight = _meanTotal;
      _meanLeft = 0.0;
      _sqSumRight = _sqSumTotal;
      _sqSumLeft = 0.0;
      _varRight = (_sqSumRight/static_cast<double>(BaseCriterion::_numberOfSamplesInThisNode) -
                   _meanRight * _meanRight);
      _varLeft = 0.0;
      _sumRight = _sumTotal;
      _sumLeft = 0.0;
    }

    void
    _update(const long newPos)
    {
      for (long pos = BaseCriterion::_currentPosition; pos < newPos; ++pos){
        long dataId = BaseCriterion::_sampleIndices->at(pos);
        double thisLabel = (*BaseCriterion::_labelData)(dataId);

        _sumLeft += thisLabel;
        _sumRight -= thisLabel;
        _sqSumLeft += thisLabel * thisLabel;
        _sqSumRight -= thisLabel*thisLabel;
      }

      _meanLeft = _sumLeft/static_cast<double>(BaseCriterion::_numberOfSamplesOnLeft);
      _meanRight = _sumRight/static_cast<double>(BaseCriterion::_numberOfSamplesOnRight);
      _varRight = (_sqSumRight/static_cast<double>(BaseCriterion::_numberOfSamplesInThisNode) -
                   _meanRight * _meanRight);
      _varLeft = (_sqSumLeft/static_cast<double>(BaseCriterion::_numberOfSamplesOnLeft) -
                  _meanLeft * _meanLeft);
      _varRight = (_sqSumRight/static_cast<double>(BaseCriterion::_numberOfSamplesOnRight) -
                   _meanRight * _meanRight);
    }

    double
    calculateNodeImpurity() const
    {
      double total = (_sqSumTotal/static_cast<double>(BaseCriterion::_numberOfSamplesInThisNode) -
                      _meanTotal*_meanTotal);
      return total;
    }

    void
    calculateChildrenImpurity(double* impurityLeft, double* impurityRight) const
    {
      *impurityLeft = _varLeft;
      *impurityRight = _varRight;
    }

    double
    nodeValue() const
    {
      return _meanTotal;
    }

  private:
    double _meanLeft;
    double _meanRight;
    double _meanTotal;
    double _sqSumLeft;
    double _sqSumRight;
    double _sqSumTotal;
    double _varLeft;
    double _varRight;
    double _sumLeft;
    double _sumRight;
    double _sumTotal;
  };
}
#endif //_REGRESSION_CRITERION
