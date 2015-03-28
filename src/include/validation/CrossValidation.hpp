#include <iostream>

namespace CPPLearn{
  template<class LearningModel>
  class CrossValidation{
  public:

    CrossValidation(LearningModel* learningModel_) : learningModel{learningModel_}{}
    VectorXd computeValidationScores(const MatrixXd& data,
                                     const VectorXd& labels,
                                     const unsigned int numberOfFolds=5,
                                     const bool randomShuffle=false){

      VectorXd scores(numberOfFolds); scores.fill(0);

      if (data.rows() != labels.size())
        throw std::runtime_error("data and labels have different length.");

      size_t numberOfData = data.rows();
      if (numberOfData < numberOfFolds)
        throw std::range_error("too few data points, even smaller than number of folds .");

      size_t numberOfFeatures=data.cols();

      size_t unitTestSize = static_cast<int>((double)numberOfData/numberOfFolds + 0.5);
      size_t firstRoundTestSize = numberOfData - unitTestSize*(numberOfFolds-1);
      size_t restDataSize = unitTestSize*(numberOfFolds-1);

      learningModel->trainModel(data.bottomRows(restDataSize), labels.tail(restDataSize));

      VectorXd predictedLabel = learningModel->predict(data.topRows(firstRoundTestSize));
      scores(0)=(predictedLabel-labels.head(firstRoundTestSize)).norm();

      MatrixXd testData = data.block(firstRoundTestSize, 0, unitTestSize, numberOfFeatures);
      VectorXd testLabel = labels.segment(firstRoundTestSize,unitTestSize);

      MatrixXd trainData = data.bottomRows(restDataSize - unitTestSize);
      VectorXd trainLabel = labels.tail(restDataSize - unitTestSize);

      size_t offset=0;
      for (size_t roundIndex=1; roundIndex<numberOfFolds; ++roundIndex){
        learningModel->trainModel(trainData, trainLabel);
        VectorXd predictedLabel = learningModel->predict(testData);
        scores(roundIndex) = (predictedLabel-testLabel).norm();

        if (roundIndex != numberOfFolds-1){
          MatrixXd tempData= trainData.block(offset, 0, unitTestSize, numberOfFeatures);
          VectorXd tempLabel= trainLabel.segment(offset, unitTestSize);

          trainData.block(offset, 0, unitTestSize, numberOfFeatures)=testData;
          trainLabel.segment(offset, unitTestSize)=testLabel;

          testData=tempData;
          testLabel=tempLabel;
          ++offset;
        }
      }
      return scores;
    };

  private:
    LearningModel* learningModel;
  };

}
