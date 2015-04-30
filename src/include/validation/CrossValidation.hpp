#include <core/Definitions.hpp>

namespace CPPLearn
{
  /**
   * enter a brief description
   */
  class CrossValidation
  {
  public:
    static vector<vector<size_t > > KFolds(const size_t numberOfData,
                                           const size_t numberOfFolds=5,
                                           const bool shuffle=false,
                                           const unsigned randomSeed=0);

    static vector<vector<size_t > > StratifiedKFolds(const Labels& labels,
                                                     const size_t numberOfFolds=5,
                                                     const bool shuffle=false,
                                                     const unsigned randomSeed=0);

    /**
     * Creates the model, with empty model initialized.
     *
     * @param
     */
    CrossValidation(const MatrixXd& data_, const Labels& labels_,
                    const unsigned int numberOfFolds_=5,
                    const bool shuffle=false) :
      data{data_}, labels{labels_}, numberOfFolds{numberOfFolds_}
    {
      size_t numberOfData = data.rows();
      if (data.rows() != labels.labelData.size()){
        throwException("Error happened in CrossValidation constructor:\n"
                       "Provided data and label sizes mismatch!\n"
                       "number of data = (%lu), number of labels = (%ld).\n",
                       numberOfData, labels.labelData.size());

      }

      if (numberOfFolds <= 1 || numberOfFolds > numberOfData){
        throwException("Error happened in CrossValidation constructor:\n"
                       "Number of folds must be greater than zero "
                       "and no larger than number of data! \n");
      }

      switch (labels.labelType){
      case ProblemType::Classification :
        try {
          foldsIndices=StratifiedKFolds(labels, numberOfFolds, shuffle);
        }
        catch(std::exception& e) {
          printf("Error happend in CrossValidation constructor.\n");
          throw;
        }
        break;
      case ProblemType::Regression :
        try{
          foldsIndices=KFolds((unsigned)labels.labelData.size(), numberOfFolds, shuffle);
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

    CrossValidation(const MatrixXd& data_, const Labels& labels_,
                    const bool shuffle) :
      CrossValidation(data_, labels_, 5, shuffle) { }

    template<class LearningModel>
    VectorXd computeValidationLosses(LearningModel* learningModel)
    {
      if (LearningModel::ModelType != labels.labelType){
        throwException("Error happened when calling computeValidationScore function:"
                       "The learning model and labels have different problem types!\n");
      }

      double (*const LossFunction)(const Labels&, const Labels&)=LearningModel::LossFunction;

      VectorXd losses(numberOfFolds);
      size_t numberOfFeatures=data.cols();
      size_t numberOfData=data.rows();

      for (auto& indicesOfThisFold : foldsIndices){
        size_t currentRoundIndex=&indicesOfThisFold-&foldsIndices[0];
        size_t numberOfTestsOfThisFold=indicesOfThisFold.size();
        size_t numberOfTrainsOfThisFold=numberOfData-numberOfTestsOfThisFold;

        MatrixXd trainDataOfThisFold(numberOfTrainsOfThisFold, numberOfFeatures);
        MatrixXd testDataOfThisFold(numberOfTestsOfThisFold, numberOfFeatures);

        Labels trainLabelsOfThisFold{labels.labelType};
        Labels testLabelsOfThisFold{labels.labelType};
        trainLabelsOfThisFold.labelData.resize(numberOfTrainsOfThisFold);
        testLabelsOfThisFold.labelData.resize(numberOfTestsOfThisFold);

        for (auto& dataIndex : indicesOfThisFold){
          size_t currentIndex=&dataIndex-&indicesOfThisFold[0];
          testDataOfThisFold.row(currentIndex)=data.row(dataIndex);
          testLabelsOfThisFold.labelData(currentIndex)=labels.labelData(dataIndex);
        }

        size_t trainDataIndex=0;
        for (auto& indicesOfOtherFold : foldsIndices){
          if (&indicesOfOtherFold==&indicesOfThisFold) continue;
          for (auto& dataIndex : indicesOfOtherFold){
            trainDataOfThisFold.row(trainDataIndex) = data.row(dataIndex);
            trainLabelsOfThisFold.labelData(trainDataIndex)= labels.labelData(dataIndex);
            ++trainDataIndex;
          }
        }

        assert(trainDataIndex==numberOfTrainsOfThisFold);

        try {
          learningModel->train(trainDataOfThisFold, trainLabelsOfThisFold);
          Labels predictedLabels = learningModel->predict(testDataOfThisFold);
          losses(currentRoundIndex)=
            LossFunction(predictedLabels, testLabelsOfThisFold)/numberOfTestsOfThisFold;
        }
        catch (std::exception& e){
          printf("Error happened in computeValidationScore function: "
                 "CV round = (%lu).\n", currentRoundIndex);
          throw;
        }
      }

      return losses;
    };

  private:
    vector<vector<size_t> > foldsIndices;
    const MatrixXd& data;
    const Labels& labels;
    const size_t numberOfFolds;
  };

  vector<vector<size_t > > CrossValidation::KFolds(const size_t numberOfData,
                                                   const size_t numberOfFolds,
                                                   const bool shuffle,
                                                   const unsigned randomSeed)
  {
    if (numberOfFolds <= 1 || numberOfFolds > numberOfData){
      throwException("Error happened in function KFold:\n"
                     "Number of folds must be greater than zero "
                     "and no larger than number of data! \n");
    }

    vector<size_t> indices; indices.reserve(numberOfData);

    for (size_t index=0; index<numberOfData; ++index) indices.push_back(index);
    if (shuffle){
      std::shuffle(indices.begin(), indices.end(), std::default_random_engine(randomSeed));
    }

    size_t meanFoldSize = numberOfData/numberOfFolds;
    vector<size_t> foldSizes(numberOfFolds, meanFoldSize);

    size_t unbalancedNumberOfFolds=numberOfData % numberOfFolds;

    for (size_t foldIndex=0; foldIndex<unbalancedNumberOfFolds; ++foldIndex)
      ++foldSizes[foldIndex];

    vector<vector<size_t> > foldsIndices;

    auto current = indices.begin();
    for (auto& foldSize : foldSizes){
      auto start=current;
      auto stop=current+foldSize;
      vector<size_t> indicesOfThisFold(start, stop);
      foldsIndices.push_back(std::move(indicesOfThisFold));
      current = stop;
    }

    return foldsIndices;
  }

  vector<vector<size_t > > CrossValidation::StratifiedKFolds(const Labels& labels,
                                                             const size_t numberOfFolds,
                                                             const bool shuffle,
                                                             const unsigned randomSeed)
  {
    size_t numberOfData=labels.labelData.size();
    const VectorXd& labelData=labels.labelData;

    if (numberOfFolds <= 1 || numberOfFolds > numberOfData){
      throwException("Error happened in function StratifiedKFold:\n"
                     "Number of folds must be greater than zero "
                     "and no larger than number of data! \n");
    }

    if (labels.labelType != ProblemType::Classification){
      throwException("Error happened in function StratifiedKFold:\n"
                     "Label type should be Classification.\n ");
    }

    //const size_t numberOfClasses = *std::max_element(&labelData[0], &labelData[numberOfData-1]);
    const size_t numberOfClasses=labelData.maxCoeff()+1;
    vector<size_t> labelsCount(numberOfClasses, 0);

    for (size_t labelIndex=0; labelIndex<numberOfData; ++ labelIndex){
      if (labelData(labelIndex) < 0 || labelData(labelIndex) >= numberOfClasses){
        throwException("Error happened in function StratifiedKFold:\n"
                       "number of classes is (%lu), "
                       "thus  I am expecting label range between (0) and (%lu)! "
                       "the (%lu)th label is (%f)!",
                       numberOfClasses, numberOfClasses-1,
                       labelIndex, labelData(labelIndex));
      }
      ++labelsCount[labelData(labelIndex)];
    }

    for (auto& labelCount : labelsCount)
      if (labelCount == 0){
        throwException("Error happened in function StratifiedKFold:\n"
                       "None of the provided training data is classified to class (%lu). "
                       "You need to consider reducing the input number of classes!\n",
                       &labelCount-&labelsCount[0]);
      }

    size_t minLabelCount = *std::min_element(labelsCount.begin(), labelsCount.end());

    if (numberOfFolds > minLabelCount){
      throwException("Error from function StratifiedKFold:\n"
                     "The least polulated label has (%lu) "
                     "members, which is too few. The minimum"
                     "number of labels for any class cannot"
                     "be less than number of folds = (%lu).",
                     minLabelCount, numberOfFolds);
    }

    vector<vector<vector<size_t> > > eachClassKFolds(numberOfClasses);
    for (size_t classId=0; classId<numberOfClasses; ++classId)
      eachClassKFolds[classId]=KFolds(labelsCount[classId], numberOfFolds,
                                      shuffle, randomSeed+classId);


    vector<vector<size_t> > eachClassIndexMapToFold;
    for (size_t classId=0; classId<numberOfClasses; ++classId){
      const vector<vector<size_t> >& thisClassKFolds = eachClassKFolds[classId];
      vector<size_t> thisClassIndexMapToFold(labelsCount[classId]);
      for (size_t foldId=0; foldId<numberOfFolds; ++foldId)
        for (auto& index : thisClassKFolds[foldId])
          thisClassIndexMapToFold[index]=foldId;
      eachClassIndexMapToFold.push_back(std::move(thisClassIndexMapToFold));
    }

    vector<size_t> testFolds(numberOfData);
    vector<size_t> foldSizes(numberOfFolds, 0);
    vector<size_t> labelsReCount(numberOfClasses, 0);

    for (size_t dataId=0; dataId<numberOfData; ++dataId){
      const size_t thisLabel= (unsigned) labelData[dataId];
      size_t foldIndex=eachClassIndexMapToFold[thisLabel][labelsReCount[thisLabel]];
      testFolds[dataId] = foldIndex;
      ++labelsReCount[thisLabel];
      ++foldSizes[foldIndex];
    }

    for (size_t classId=0; classId<numberOfClasses; ++classId)
      assert(labelsReCount[classId]==labelsCount[classId]);

    vector<vector<size_t> > foldsIndices;
    for (size_t foldId=0; foldId<numberOfFolds; ++foldId){
      vector<size_t> indicesOfThisFold;
      indicesOfThisFold.reserve(foldSizes[foldId]);
      foldsIndices.push_back(std::move(indicesOfThisFold));
    }

    for (size_t dataId=0; dataId<numberOfData; ++dataId){
      size_t foldIndex=testFolds[dataId];
      foldsIndices[foldIndex].push_back(dataId);
    }

    return foldsIndices;
  }
}
