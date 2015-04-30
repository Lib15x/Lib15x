#ifndef MODEL_MULTICLASS_CASSIFIER
#define MODEL_MULTICLASS_CASSIFIER

#include <core/Definitions.hpp>

namespace CPPLearn
{
  namespace Models
  {
    /**
     * enter a brief description
     */
    template<class BinaryClassifier>
    class MulticlassClassifier
    {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="MultiClassClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        BinaryClassifier::LossFunction;

      typedef std::unique_ptr<BinaryClassifier> ModelPtr;

      /**
       * Creates the model, with empty model initialized.
       *
       * @param
       */
      MulticlassClassifier(size_t numberOfClasses_, BinaryClassifier binaryModel_) :
        numberOfFeatures{binaryModel_.getNumberOfFeatures()},
        numberOfClasses{numberOfClasses_},
        binaryModels{numberOfClasses*(numberOfClasses-1)/2}
      {
        for (auto& binaryModel : binaryModels)
          binaryModel = std::make_unique<BinaryClassifier>(binaryModel_);
      }

      MulticlassClassifier(size_t numberOfClasses_, vector<BinaryClassifier> binaryModels_) :
        numberOfClasses{numberOfClasses_},
        numberOfFeatures(binaryModels_[0].getNumberOfFeatures()),
        binaryModels(numberOfClasses*(numberOfClasses-1)/2)
      {
        for (auto& binaryModel : binaryModels)
          binaryModel = std::make_unique<BinaryClassifier>(binaryModels_[index]);
      }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels.labelType != ProblemType::Classification){
          throwException("Error happen when training LibSVM model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels.labelData;

        if ((unsigned)trainData.cols() != numberOfFeatures){
          throwException("Error happen when training multiclass classifier: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("Error happen when training multiclass classifier: "
                         "data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }

        vector<size_t> labelsCount(numberOfClasses, 0);
        const size_t numberOfTrainData = trainData.rows();

        for (size_t labelIndex=0; labelIndex<numberOfTrainData; ++ labelIndex){
          if (labelData(labelIndex) < 0 || labelData(labelIndex) >= numberOfClasses){
            throwException("Error happen when training multiclass classifier: "
                           "number of classes is (%lu), "
                           "thus  I am expecting label range between (0) and (%lu)! "
                           "the (%lu)th label is (%f)!",
                           numberOfClasses, numberOfClasses-1,
                           labelIndex, labelData(labelIndex));
          }
          ++labelsCount[labelData(labelIndex)];
        }

        for (size_t classIndex=0; classIndex<numberOfClasses; ++classIndex)
          if (labelsCount[classIndex] == 0){
            throwException("Error happened from Multiclass classifer:\n"
                           "None of the provided training data is classified to class (%lu). "
                           "You need to consider reducing the input number of classes!\n",
                           classIndex);
          }

        if(verbose == VerboseFlag::Verbose){
          cout<<"In Mulicalss Classifier, Multiclass training input label summary: "<<endl;
          cout<<"number of classes: "<<numberOfClasses<<endl;
          cout<<"number of training data: "<<numberOfTrainData<<endl;
          for (size_t classIndex=0; classIndex<numberOfClasses; ++classIndex)
            cout<<"number of training data labeled "<<classIndex<<": "<<
              labelsCount[classIndex]<<endl;
        }

        size_t modelCount=0;
        for (size_t firstIndex=0; firstIndex<numberOfClasses-1; ++firstIndex)
          for (size_t secondIndex=firstIndex+1; secondIndex<numberOfClasses; ++secondIndex){
            size_t numberOfLabels=labelsCount[firstIndex]+labelsCount[secondIndex];

            Labels binaryLabels{BinaryClassifier::ModelType};
            binaryLabels.labelData.resize(numberOfLabels);
            MatrixXd dataForThisRound(numberOfLabels, numberOfFeatures);
            size_t subModelDataCount=0;
            for (size_t dataIndex=0; dataIndex<numberOfTrainData; ++dataIndex){
              if (labelData(dataIndex)==firstIndex){
                binaryLabels.labelData(subModelDataCount)=0;
                dataForThisRound.row(subModelDataCount)=trainData.row(dataIndex);
                ++subModelDataCount;
                continue;
              }
              if (labelData(dataIndex)==secondIndex){
                binaryLabels.labelData(subModelDataCount)=1;
                dataForThisRound.row(subModelDataCount)=trainData.row(dataIndex);
                ++subModelDataCount;
                continue;
              }
            }
            assert(subModelDataCount==numberOfLabels);
            binaryModels[modelCount]->train(dataForThisRound, binaryLabels);
            ++modelCount;
          }

        assert(modelCount==numberOfClasses*(numberOfClasses-1)/2);

        modelTrained=true;
      }

      /**
       * Calculate predictions based on trained model, returns the predicted
       * labels. The model has to be trained first.
       *
       * @param testData predictors, the number of columns should be the
       * same as number of features.
       */
      Labels predict(const MatrixXd& testData) const
      {
        if (!modelTrained){
          throwException("Error happen when predicting with multiclass classifier: "
                         "model has not been trained yet!");
        }

        if ((unsigned)testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting with multiclass classifier: "
                         "Invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, testData.cols());
        }

        size_t numberOfTests=testData.rows();
        Labels predictedLabels(ModelType);
        predictedLabels.labelData.resize(numberOfTests);

        for (size_t testIndex=0; testIndex<numberOfTests; ++testIndex){
          VectorXd oneInstance=testData.row(testIndex);
          vector<size_t> votes(numberOfClasses, 0);
          size_t modelCount=0;
          for (size_t firstIndex=0; firstIndex<numberOfClasses-1; ++firstIndex)
            for (size_t secondIndex=firstIndex+1; secondIndex<numberOfClasses; ++secondIndex){
              double result = binaryModels[modelCount]->predictOne(oneInstance);
              result == 0 ? ++votes[firstIndex] : ++votes[secondIndex];
              ++modelCount;
            }

          assert(modelCount==numberOfClasses*(numberOfClasses-1)/2);

          auto maxpos=std::max_element(votes.begin(), votes.end());
          auto it=std::find(maxpos+1, votes.end(), *maxpos);
          if (it !=votes.end())
            printf("Warning: In multiclass prediction, the choice predict to label "
                   "for test data NO. (%lu) is not unique. "
                   "I choose to predict label (%lu) since it is smaller in numeric order.\n",
                   testIndex, maxpos-votes.begin());
          predictedLabels.labelData(testIndex)=maxpos-votes.begin();
        }

        return predictedLabels;
      }

      /**
       * Clear the model.
       */
      void clear()
      {
        for (size_t classIndex=0; classIndex<binaryModels.size(); ++classIndex)
          binaryModels[classIndex]->clear();

        modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return verbose;
      }

      /**
       * Destructor.
       */
      ~MulticlassClassifier(){}

    private:
      const size_t numberOfFeatures;
      const size_t numberOfClasses;
      vector<ModelPtr> binaryModels;
      bool modelTrained=false;
      VerboseFlag verbose = VerboseFlag::Quiet;
    };
  }
}

#endif // MODEL_MULTICLASS_CASSIFIER
