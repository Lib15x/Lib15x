#ifndef CROSS_VALIDATION
#define CROSS_VALIDATION
#include "../core/Definitions.hpp"

namespace CPPLearn
{
  /**
   * enter a brief description
   */
  class CrossValidation
  {
  public:
    static vector<vector<long> >
    KFolds(const long numberOfData, const long numberOfFolds=5,
           const bool shuffle=false, const long randomSeed=0);

    static vector<vector<long> >
    StratifiedKFolds(const Labels& labels, const long numberOfFolds=5,
                     const bool shuffle=false, const long randomSeed=0);

    /**
     * Creates the model, with empty model initialized.
     *
     * @param
     */
    CrossValidation(const MatrixXd& data, const Labels& labels,
                    const long numberOfFolds=5,
                    const bool shuffle=false) :
      _data{data}, _labels{labels}, _numberOfFolds{numberOfFolds}
    {
      long numberOfData = _data.rows();
      if (_data.rows() != _labels._labelData.size()){
        throwException("Error happened in CrossValidation constructor:\n"
                       "Provided data and label sizes mismatch!\n"
                       "number of data = (%ld), number of labels = (%ld).\n",
                       numberOfData, _labels._labelData.size());

      }

      if (_numberOfFolds <= 1 || _numberOfFolds > numberOfData){
        throwException("Error happened in CrossValidation constructor:\n"
                       "Number of folds must be greater than zero "
                       "and no larger than number of data! \n");
      }

      switch (_labels._labelType){
      case ProblemType::Classification :
        try {
          _foldsIndices=StratifiedKFolds(_labels, _numberOfFolds, shuffle);
        }
        catch(std::exception& e) {
          printf("Error happend in CrossValidation constructor.\n");
          throw;
        }
        break;
      case ProblemType::Regression :
        try{
          _foldsIndices=KFolds(_labels._labelData.size(), _numberOfFolds, shuffle);
        }
        catch (std::exception& e){
          printf("Error happend in CrossValidation constructor.\n");
          throw;
        }
        break;
      default:
        {
          throwException("Error happened in CrossValidation constructor:\n"
                         "the input label type mush be either Classification or Regression!\n");
        }
      }
    }

    CrossValidation(const MatrixXd& data, const Labels& labels,
                    const bool shuffle) :
      CrossValidation(data, labels, 5, shuffle) { }

    template<class LearningModel>
    VectorXd
    computeValidationLosses(LearningModel* learningModel) const
    {
      if (LearningModel::ModelType != _labels._labelType) {
        throwException("Error happened when calling computeValidationScore function:"
                       "The learning model and labels have different problem types!\n");
      }
      double (*const LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;
      VectorXd losses{_numberOfFolds}; losses.fill(0);
      long numberOfFeatures=_data.cols();
      long numberOfData=_data.rows();

      for (long currentRoundId =0; currentRoundId<_numberOfFolds; ++currentRoundId) {
        const vector<long>& indicesOfThisFold = _foldsIndices[currentRoundId];
        long numberOfTestsOfThisFold = static_cast<long>(indicesOfThisFold.size());
        long numberOfTrainsOfThisFold=numberOfData-numberOfTestsOfThisFold;

        MatrixXd trainDataOfThisFold{numberOfTrainsOfThisFold, numberOfFeatures};
        MatrixXd testDataOfThisFold{numberOfTestsOfThisFold, numberOfFeatures};

        Labels trainLabelsOfThisFold{_labels._labelType};
        Labels testLabelsOfThisFold{_labels._labelType};
        trainLabelsOfThisFold._labelData.resize(numberOfTrainsOfThisFold);
        testLabelsOfThisFold._labelData.resize(numberOfTestsOfThisFold);

        for (long id=0; id<numberOfTestsOfThisFold; ++id){
          testDataOfThisFold.row(id) = _data.row(indicesOfThisFold[id]);
          testLabelsOfThisFold._labelData(id) = _labels._labelData(indicesOfThisFold[id]);
        }

        long trainDataIndex=0;
        for (const auto& indicesOfOtherFold : _foldsIndices){
          if (&indicesOfOtherFold==&indicesOfThisFold) continue;
          for (auto& dataIndex : indicesOfOtherFold){
            trainDataOfThisFold.row(trainDataIndex) = _data.row(dataIndex);
            trainLabelsOfThisFold._labelData(trainDataIndex)= _labels._labelData(dataIndex);
            ++trainDataIndex;
          }
        }

        assert(trainDataIndex==numberOfTrainsOfThisFold);

        try {
          learningModel->train(trainDataOfThisFold, trainLabelsOfThisFold);
          Labels predictedLabels = learningModel->predict(testDataOfThisFold);
          losses(currentRoundId)=
            LossFunction(predictedLabels, testLabelsOfThisFold)/static_cast<double>(numberOfTestsOfThisFold);
        }
        catch (std::exception& e){
          printf("Error happened in computeValidationScore function: "
                 "CV round = (%ld).\n", currentRoundId);
          throw;
        }
      }

      return losses;
    };

  private:
    vector<vector<long> > _foldsIndices;
    const MatrixXd& _data;
    const Labels& _labels;
    const long _numberOfFolds;
  };

  vector<vector<long > >
  CrossValidation::KFolds(const long numberOfData, const long numberOfFolds,
                          const bool shuffle, const long randomSeed)
  {
    if (numberOfFolds <= 1 || numberOfFolds > numberOfData){
      throwException("Error happened in function KFold:\n"
                     "Number of folds must be greater than zero "
                     "and no larger than number of data! \n");
    }

    vector<long> indices; indices.reserve(numberOfData);

    for (long index=0; index<numberOfData; ++index) indices.push_back(index);
    if (shuffle){
      std::shuffle(std::begin(indices), std::end(indices), std::default_random_engine(randomSeed));
    }

    long meanFoldSize = numberOfData/numberOfFolds;
    vector<long> foldSizes(numberOfFolds, meanFoldSize);

    long unbalancedNumberOfFolds=numberOfData % numberOfFolds;

    std::transform(std::begin(foldSizes), std::begin(foldSizes)+unbalancedNumberOfFolds,
                   std::begin(foldSizes), [](long i){return ++i;});

    vector<vector<long> > foldsIndices;

    auto current = std::begin(indices);
    for (const auto& foldSize : foldSizes){
      auto start=current;
      auto stop=current+foldSize;
      foldsIndices.emplace_back(start, stop);
      current = stop;
    }

    return foldsIndices;
  }

  vector<vector<long > >
  CrossValidation::StratifiedKFolds(const Labels& labels, const long numberOfFolds,
                                    const bool shuffle, const long randomSeed)
  {
    const long numberOfData=labels._labelData.size();
    const VectorXd& labelData=labels._labelData;

    if (numberOfFolds <= 1 || numberOfFolds > numberOfData){
      throwException("Error happened in function StratifiedKFold:\n"
                     "Number of folds must be greater than zero "
                     "and no larger than number of data! \n");
    }

    if (labels._labelType != ProblemType::Classification){
      throwException("Error happened in function StratifiedKFold:\n"
                     "Label type should be Classification.\n ");
    }

    const long numberOfClasses=static_cast<long>(labelData.maxCoeff())+1;
    vector<long> labelsCount(numberOfClasses, 0);

    for (long labelIndex=0; labelIndex<numberOfData; ++labelIndex){
      if (labelData(labelIndex) < 0 || labelData(labelIndex) >= numberOfClasses){
        throwException("Error happened in function StratifiedKFold:\n"
                       "number of classes is (%ld), "
                       "thus  I am expecting label range between (0) and (%ld)! "
                       "the (%ld)th label is (%f)!",
                       numberOfClasses, numberOfClasses-1,
                       labelIndex, labelData(labelIndex));
      }
      ++labelsCount[static_cast<long>(labelData(labelIndex))];
    }

    for (auto& labelCount : labelsCount)
      if (labelCount == 0){
        throwException("Error happened in function StratifiedKFold:\n"
                       "None of the provided training data is classified to class (%ld). "
                       "You need to consider reducing the input number of classes!\n",
                       &labelCount-&labelsCount[0]);
      }

    const auto minLabelCount = *std::min_element(std::begin(labelsCount), std::end(labelsCount));

    if (numberOfFolds > minLabelCount){
      throwException("Error from function StratifiedKFold:\n"
                     "The least polulated label has (%ld) "
                     "members, which is too few. The minimum"
                     "number of labels for any class cannot"
                     "be less than number of folds = (%ld).",
                     minLabelCount, numberOfFolds);
    }

    vector<vector<vector<long> > > eachClassKFolds{static_cast<unsigned>(numberOfClasses)};

    for (long classId = 0; classId<numberOfClasses; ++classId)
      eachClassKFolds[classId] = KFolds(labelsCount[classId], numberOfFolds,
                                        shuffle, randomSeed + classId);

    vector<vector<long> > eachClassIndexMapToFold;
    for (long classId=0; classId<numberOfClasses; ++classId){
      const vector<vector<long> >& thisClassKFolds = eachClassKFolds[classId];
      vector<long> thisClassIndexMapToFold(labelsCount[classId]);
      for (long foldId=0; foldId<numberOfFolds; ++foldId)
        for (const auto& index : thisClassKFolds[foldId])
          thisClassIndexMapToFold[index]=foldId;
      eachClassIndexMapToFold.push_back(std::move(thisClassIndexMapToFold));
    }

    vector<long> testFolds(numberOfData);
    vector<long> foldSizes(numberOfFolds, 0);
    vector<long> labelsReCount(numberOfClasses, 0);

    for (long dataId=0; dataId<numberOfData; ++dataId){
      const long thisLabel = static_cast<long>(labelData[dataId]);
      const long foldIndex=eachClassIndexMapToFold[thisLabel][labelsReCount[thisLabel]];
      testFolds[dataId] = foldIndex;
      ++labelsReCount[thisLabel];
      ++foldSizes[foldIndex];
    }

    for (long classId=0; classId<numberOfClasses; ++classId)
      assert(labelsReCount[classId]==labelsCount[classId]);

    vector<vector<long> > foldsIndices;
    for (long foldId=0; foldId<numberOfFolds; ++foldId){
      vector<long> indicesOfThisFold;
      indicesOfThisFold.reserve(foldSizes[foldId]);
      foldsIndices.push_back(std::move(indicesOfThisFold));
    }

    for (long dataId=0; dataId<numberOfData; ++dataId){
      long foldIndex=testFolds[dataId];
      foldsIndices[foldIndex].push_back(dataId);
    }

    return foldsIndices;
  }
}

#endif //CROSS_VALIDATION
