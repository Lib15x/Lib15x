#ifndef _CLASSIFICATION_CRITERION
#define _CLASSIFICATION_CRITERION

#include "../core/Definitions.hpp"
#include "./_BaseCriterion.hpp"

namespace CPPLearn {

  double
  bernoulli(const vector<long>& labelsCount) {
    const long numberOfSamples =
      std::accumulate(std::begin(labelsCount), std::end(labelsCount), 0);

    double impurity = 1.0;
    for (auto thisLabelCount : labelsCount)
      impurity *= static_cast<double>(thisLabelCount)/static_cast<double>(numberOfSamples);

    return impurity;
  }

  double
  gini(const vector<long>& labelsCount) {
    const long numberOfSamples =
      std::accumulate(std::begin(labelsCount), std::end(labelsCount), 0);

    double impurity = 1.0;
    for (auto thisLabelCount : labelsCount) {
      double temp = static_cast<double>(thisLabelCount)/static_cast<double>(numberOfSamples);
      impurity -= temp*temp;
    }

    return impurity;
  }

  double
  entropy(const vector<long>& labelsCount) {
    const long numberOfSamples =
      std::accumulate(std::begin(labelsCount), std::end(labelsCount), 0);

    double impurity = 0.0;
    for (auto thisLabelCount : labelsCount)
      if (thisLabelCount > 0){
        double temp = static_cast<double>(thisLabelCount)/static_cast<double>(numberOfSamples);
        impurity -= temp*log(temp);
      }

    return impurity;
  }

  template<double (*ImpurityRule)(const vector<long>&)>
  class _ClassificationCriterion :
    public _BaseCriterion<_ClassificationCriterion<ImpurityRule> > {
  public:
    using BaseCriterion = _BaseCriterion<_ClassificationCriterion>;

    explicit _ClassificationCriterion(const VectorXd* labelData, const long numberOfClasses) :
      BaseCriterion{labelData}, _numberOfClasses{numberOfClasses},
      _labelsCountTotal(numberOfClasses,0),
      _labelsCountLeft(numberOfClasses,0), _labelsCountRight(numberOfClasses,0) { }

    void
    _init()
    {
      long startIndex = BaseCriterion::_startIndex;
      long endIndex = BaseCriterion::_endIndex;

      std::fill(std::begin(_labelsCountTotal), std::end(_labelsCountTotal), 0);
      for (long sampleId = startIndex; sampleId < endIndex; ++sampleId) {
        long dataId = BaseCriterion::_sampleIndices->at(sampleId);
        long thisLabel =  static_cast<long>((*BaseCriterion::_labelData)(dataId));
        ++_labelsCountTotal[thisLabel];
      }
    }

    void
    _reset()
    {
      std::copy(std::begin(_labelsCountTotal), std::end(_labelsCountTotal),
                std::begin(_labelsCountRight));
      std::fill(std::begin(_labelsCountLeft), std::end(_labelsCountLeft), 0);
    }

    void
    _update(const long newPos)
    {
      for (long pos = BaseCriterion::_currentPosition; pos < newPos; ++pos){
        long dataId = BaseCriterion::_sampleIndices->at(pos);
        long thisLabel = static_cast<long>((*BaseCriterion::_labelData)(dataId));
        ++_labelsCountLeft[thisLabel];
        --_labelsCountRight[thisLabel];
      }
    }

    double
    calculateNodeImpurity() const
    {
      return ImpurityRule(_labelsCountTotal);
    }

    void
    calculateChildrenImpurity(double* impurityLeft,
                              double* impurityRight) const
    {
      *impurityLeft = ImpurityRule(_labelsCountLeft);
      *impurityRight = ImpurityRule(_labelsCountRight);
    }

    const vector<long>&
    nodeValue() const
    {
      return _labelsCountTotal;
    }

  private:
    long _numberOfClasses;
    vector<long> _labelsCountTotal;
    vector<long> _labelsCountLeft;
    vector<long> _labelsCountRight;
  };
}
#endif //_CLASSIFICATION_CRITERION
