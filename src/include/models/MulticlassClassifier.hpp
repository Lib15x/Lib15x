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
      static constexpr const char* ModelName = "MultiClassClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        BinaryClassifier::LossFunction;
      using ModelPtr = std::unique_ptr<BinaryClassifier>;

      /**
       * Creates the model, with empty model initialized.
       *
       * @param
       */
      MulticlassClassifier(const long numberOfClasses, const BinaryClassifier& binaryModel) :
        _numberOfFeatures{binaryModel.getNumberOfFeatures()},
        _numberOfClasses{numberOfClasses},
        _binaryModels{static_cast<unsigned>(_numberOfClasses*(_numberOfClasses-1)/2)}
      {
        static_assert(BinaryClassifier::ModelType == ModelType,
                      "modelType should be classification");
        for (auto& model : _binaryModels)
          model = std::make_unique<BinaryClassifier>(binaryModel);
      }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void
      train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Classification){
          throwException("Error happen when training LibSVM model: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training multiclass classifier: "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("Error happen when training multiclass classifier: "
                         "data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }

        vector<long> labelsCount(_numberOfClasses, 0);
        const long numberOfTrainData = trainData.rows();

        for (long labelIndex=0; labelIndex<numberOfTrainData; ++labelIndex){
          if (labelData(labelIndex) < 0 || labelData(labelIndex) >= _numberOfClasses){
            throwException("Error happen when training multiclass classifier: "
                           "number of classes is (%ld), "
                           "thus  I am expecting label range between (0) and (%ld)! "
                           "the (%ld)th label is (%f)!",
                           _numberOfClasses, _numberOfClasses-1,
                           labelIndex, labelData(labelIndex));
          }
          ++labelsCount[static_cast<long>(labelData(labelIndex))];
        }

        for (long classIndex=0; classIndex<_numberOfClasses; ++classIndex)
          if (labelsCount[classIndex] == 0){
            throwException("Error happened from Multiclass classifer:\n"
                           "None of the provided training data is classified to class (%ld). "
                           "You need to consider reducing the input number of classes!\n",
                           classIndex);
          }

        if(_verbose == VerboseFlag::Verbose){
          cout<<"In Mulicalss Classifier, Multiclass training input label summary: "<<endl;
          cout<<"number of classes: "<<_numberOfClasses<<endl;
          cout<<"number of training data: "<<numberOfTrainData<<endl;
          for (long classIndex=0l; classIndex<_numberOfClasses; ++classIndex)
            cout<<"number of training data labeled "<<classIndex<<": "<<
              labelsCount[classIndex]<<endl;
        }

        long modelCount=0;
        for (long firstIndex=0; firstIndex<_numberOfClasses-1; ++firstIndex)
          for (long secondIndex=firstIndex+1; secondIndex<_numberOfClasses; ++secondIndex){
            long numberOfLabels=labelsCount[firstIndex] + labelsCount[secondIndex];
            Labels binaryLabels{BinaryClassifier::ModelType};
            binaryLabels._labelData.resize(numberOfLabels);
            MatrixXd dataForThisRound{numberOfLabels, _numberOfFeatures};
            long subModelDataCount=0;
            for (long dataIndex=0; dataIndex<numberOfTrainData; ++dataIndex){
              if (labelData(dataIndex) == firstIndex){
                binaryLabels._labelData(subModelDataCount)=0.0;
                dataForThisRound.row(subModelDataCount)=trainData.row(dataIndex);
                ++subModelDataCount;
                continue;
              }
              if (labelData(dataIndex)==secondIndex){
                binaryLabels._labelData(subModelDataCount)=1;
                dataForThisRound.row(subModelDataCount)=trainData.row(dataIndex);
                ++subModelDataCount;
                continue;
              }
            }
            assert(subModelDataCount==numberOfLabels);
            _binaryModels[modelCount]->train(dataForThisRound, binaryLabels);
            ++modelCount;
          }

        assert(modelCount==_numberOfClasses*(_numberOfClasses-1)/2);

        _modelTrained=true;
      }

      /**
       * Calculate predictions based on trained model, returns the predicted
       * labels. The model has to be trained first.
       *
       * @param testData predictors, the number of columns should be the
       * same as number of features.
       */
      Labels
      predict(const MatrixXd& testData) const
      {
        if (!_modelTrained){
          throwException("Error happen when predicting with multiclass classifier: "
                         "model has not been trained yet!");
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with multiclass classifier: "
                         "Invalid inpute data: "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, testData.cols());
        }

        long numberOfTests = testData.rows();
        Labels predictedLabels{ModelType};
        predictedLabels._labelData.resize(numberOfTests);

        for (long testIndex=0; testIndex<numberOfTests; ++testIndex){
          VectorXd oneInstance = testData.row(testIndex);
          vector<long> votes(_numberOfClasses, 0);
          long modelCount=0;
          for (long firstIndex=0; firstIndex<_numberOfClasses-1; ++firstIndex)
            for (long secondIndex=firstIndex+1; secondIndex<_numberOfClasses; ++secondIndex){
              double result = _binaryModels[modelCount]->predictOne(oneInstance);
              result == 0 ? ++votes[firstIndex] : ++votes[secondIndex];
              ++modelCount;
            }

          assert(modelCount==_numberOfClasses*(_numberOfClasses-1)/2);

          auto maxpos=std::max_element(votes.begin(), votes.end());
          auto it=std::find(maxpos+1, votes.end(), *maxpos);
          if (it !=std::end(votes))
            printf("Warning: In multiclass prediction, the choice predict to label "
                   "for test data NO. (%ld) is not unique. "
                   "I choose to predict label (%ld) since it is smaller in numeric order.\n",
                   testIndex, maxpos-votes.begin());
          predictedLabels._labelData(testIndex)=static_cast<double>(maxpos-begin(votes));
        }

        return predictedLabels;
      }

      /**
       * Clear the model.
       */
      void clear()
      {
        for (auto& binaryModel : _binaryModels)
          binaryModel->clear();
        _modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return _verbose;
      }

    private:
      const long _numberOfFeatures;
      const long _numberOfClasses;
      vector<ModelPtr> _binaryModels;
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
    };
  }
}

#endif // MODEL_MULTICLASS_CASSIFIER
