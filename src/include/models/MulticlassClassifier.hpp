#ifndef MODEL_MULTICLASS_CASSIFIER
#define MODEL_MULTICLASS_CASSIFIER

#include <core/Definitions.hpp>
#include "../internal/_BaseClassifier.hpp"

namespace Lib15x
{
  namespace Models
  {
    /**
     * enter a brief description
     */
    template<class BinaryClassifier>
    class MulticlassClassifier :
      public _BaseClassifier<MulticlassClassifier<BinaryClassifier> > {
    public:
      using BaseClassifier = _BaseClassifier<MulticlassClassifier<BinaryClassifier> >;
      using BaseClassifier::train;
      static constexpr const char* ModelName = "MultiClassClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&) =
        BaseClassifier::LossFunction;

      /**
       * Creates the model, with empty model initialized.
       *
       * @param
       */
      MulticlassClassifier(const long numberOfFeatures, const long numberOfClasses,
                           const BinaryClassifier& binaryModel) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _binaryModels{static_cast<unsigned>(numberOfClasses*(numberOfClasses-1)/2), binaryModel}
      {
        static_assert(BinaryClassifier::ModelType == BaseClassifier::ModelType,
                      "Binary model should be a classification model");

        if (binaryModel.getNumberOfClasses() != 2){
          throwException("Error happened when constructing Multiclass classifer:\n"
                         "the number of classes in the provided sample binary "
                         "classifier is (%ld), which is required to be 2.\n ",
                         binaryModel.getNumberOfClasses());
        }

      }

      template<typename... Args>
      MulticlassClassifier(const long numberOfFeatures, const long numberOfClasses,
                           const Args... args) :
        BaseClassifier{numberOfFeatures, numberOfClasses},
        _binaryModels{static_cast<unsigned>(numberOfClasses*(numberOfClasses-1)/2),
            BinaryClassifier{numberOfFeatures, 2, args...}}
      {
        static_assert(BinaryClassifier::ModelType == BaseClassifier::ModelType,
                      "Binary model should be a classification model");
      }


      void
      train(const MatrixXd& trainData, const Labels& trainLabels, const VectorXd& weights)
      {
        assert(weights.size()==trainLabels.size());
        long numberOfData=trainLabels.size();
        const long numberOfClasses = BaseClassifier::_numberOfClasses;
        const VectorXd& labelData=trainLabels._labelData;

        vector<long> labelsCount(numberOfClasses, 0);
        for (auto dataId=0; dataId<numberOfData; ++dataId)
          if (weights(dataId)!=0)
            ++labelsCount[static_cast<long>(labelData(dataId))];

        for (long classIndex=0; classIndex<numberOfClasses; ++classIndex)
          if (labelsCount[classIndex] == 0){
            throwException("Error happened from Multiclass classifer:\n"
                           "None of the provided training data is classified to class (%ld). "
                           "You need to consider reducing the input number of classes!\n",
                           classIndex);
          }

        long modelCount=0;
        for (long firstIndex=0; firstIndex<numberOfClasses-1; ++firstIndex)
          for (long secondIndex = firstIndex+1; secondIndex<numberOfClasses; ++secondIndex) {
            Labels binaryLabels{BinaryClassifier::ModelType};
            binaryLabels._labelData.resize(trainData.rows());
            VectorXd weightsForThisRound(numberOfData); weightsForThisRound.fill(0);
            for (long dataId=0; dataId<numberOfData; ++dataId) {
              if (labelData(dataId) == firstIndex){
                binaryLabels._labelData(dataId)=0.0;
                weightsForThisRound(dataId)=weights(dataId);
                continue;
              }
              if (labelData(dataId)==secondIndex){
                binaryLabels._labelData(dataId)=1;
                weightsForThisRound(dataId)=weights(dataId);
                continue;
              }
            }
            _binaryModels[modelCount].train(trainData, binaryLabels, weightsForThisRound);
            ++modelCount;
          }

        assert(modelCount==numberOfClasses*(numberOfClasses-1)/2);
        BaseClassifier::_modelTrained = true;
      }


      double
      predictOne(const VectorXd& instance) const
      {
        const long numberOfClasses = BaseClassifier::_numberOfClasses;
        vector<long> votes(numberOfClasses, 0);
        long modelCount=0;
        for (long firstIndex=0; firstIndex<numberOfClasses-1; ++firstIndex)
          for (long secondIndex=firstIndex+1; secondIndex<numberOfClasses; ++secondIndex){
            double result = _binaryModels[modelCount].predictOne(instance);
            result == 0 ? ++votes[firstIndex] : ++votes[secondIndex];
            ++modelCount;
          }

        assert(modelCount==numberOfClasses*(numberOfClasses-1)/2);

        auto maxpos=std::max_element(std::begin(votes), std::end(votes));
        auto it=std::find(maxpos+1, std::end(votes), *maxpos);
        if (it !=std::end(votes))
          printf("Warning: In multiclass prediction, the choice predict to label "
                 "for test data  might not be unique. "
                 "I choose to predict label (%ld) since it is smaller in numeric order.\n",
                 maxpos-votes.begin());

        return static_cast<double>(maxpos-std::begin(votes));
      }

      void _clearModel()
      {
        for (auto& binaryModel : _binaryModels)
          binaryModel.clear();
      }

    private:
      vector<BinaryClassifier> _binaryModels;
    };
  }
}

#endif // MODEL_MULTICLASS_CASSIFIER
