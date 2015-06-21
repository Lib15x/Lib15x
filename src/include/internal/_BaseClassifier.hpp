#ifndef BASE_CLASSIFIER
#define BASE_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template<class DerivedClassifier>
    class _BaseClassifier {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName=DerivedClassifier::ModelName;
      static double
      LossFunction(const Labels& predictedLabels, const Labels& testLabels)
      {
        if (predictedLabels._labelType != ProblemType::Classification ||
            testLabels._labelType != ProblemType::Classification) {
          throwException("Error happen when computing classsification error: "
                         "Input labelType must be Classification!\n");
        }

        const VectorXd& predictedLabelData=predictedLabels._labelData;
        const VectorXd& testLabelData=testLabels._labelData;

        if (predictedLabelData.size() != testLabelData.size()){
          throwException("Error happen when computing classsification error: "
                         "The inpute two labels have different sizes. "
                         "sizes of the predicted labels: (%ld); "
                         "sizes of the test labels: (%ld).\n",
                         predictedLabelData.size(), testLabelData.size());
        }

        long numberOfData=predictedLabelData.size();

        double loss=0;
        for (long dataId=0; dataId<numberOfData; ++dataId)
          if (std::lround(predictedLabelData(dataId))>=0 && std::lround(testLabelData(dataId)) >=0)
            loss+=static_cast<double>(predictedLabelData(dataId)!=testLabelData(dataId));

        return loss;
      }

      _BaseClassifier(const long numberOfFeatures, const long numberOfClasses) :
        _numberOfFeatures{numberOfFeatures}, _numberOfClasses{numberOfClasses} { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ModelType){
          throwException("Error happen when training %s model: "
                         "Input labelType must be Classification!\n", ModelName);
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training %s model, "
                         "invalid inpute data: "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         ModelName, _numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }


        vector<long> labelsCount(_numberOfClasses, 0);
        const long numberOfTrainData = trainData.rows();

        for (long labelIndex=0; labelIndex<numberOfTrainData; ++labelIndex){
          if (labelData(labelIndex) < 0 || labelData(labelIndex) >= _numberOfClasses){
            throwException("Error happen when training %s: "
                           "number of classes is (%ld), "
                           "thus  I am expecting label range between (0) and (%ld)! "
                           "the (%ld)th label is (%f)!",
                           ModelName, _numberOfClasses, _numberOfClasses-1,
                           labelIndex, labelData(labelIndex));
          }
          ++labelsCount[static_cast<long>(labelData(labelIndex))];
        }

        for (long classIndex=0; classIndex<_numberOfClasses; ++classIndex)
          if (labelsCount[classIndex] == 0){
            throwException("Error happened from Tree classifer:\n"
                           "None of the provided training data is classified to class (%ld). "
                           "You need to consider reducing the input number of classes!\n",
                           classIndex);
          }

        if(_verbose == VerboseFlag::Verbose){
          printf("In %s, training input label summary:\n", ModelName);
          cout<<"number of classes: "<<_numberOfClasses<<endl;
          cout<<"number of training data: "<<numberOfTrainData<<endl;
          for (long classIndex=0; classIndex<_numberOfClasses; ++classIndex)
            cout<<"number of training data labeled "<<classIndex<<": "<<
              labelsCount[classIndex]<<endl;
        }
        vector<long> indicesForAllTrainData(trainData.rows());
        std::iota(std::begin(indicesForAllTrainData), std::end(indicesForAllTrainData), 0);
        static_cast<DerivedClassifier*>(this)->train(trainData, trainLabels,
                                                     std::move(indicesForAllTrainData));
      }

      Labels
      predict(const MatrixXd& testData) const
      {
        if (!_modelTrained) {
          throwException("Error happen when predicting with %s model: "
                         "Model has not been trained yet!", ModelName);
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with %s model: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         ModelName, _numberOfFeatures, testData.cols());
        }

        vector<long> indicesForAllTestData(testData.rows());
        std::iota(std::begin(indicesForAllTestData), std::end(indicesForAllTestData), 0);
        Labels predictedLabels =
          predict(testData, std::move(indicesForAllTestData));

        return predictedLabels;
      }

      Labels
      predict(const MatrixXd& testData, const vector<long>& testIndices) const
      {
        assert(_modelTrained);
        Labels predictedLabels{ProblemType::Classification};
        predictedLabels._labelData.resize(testData.rows());
        predictedLabels._labelData.fill(-1.0);

        for (auto testDataId : testIndices){
          Map<const VectorXd> instance(&testData(testDataId, 0), _numberOfFeatures);
          predictedLabels._labelData(testDataId) =
            static_cast<const DerivedClassifier*>(this)->predictOne(instance);
        }

        return predictedLabels;
      }

      const long&
      getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }

      const long&
      getNumberOfClasses() const
      {
        return _numberOfClasses;
      }

      VerboseFlag&
      whetherVerbose()
      {
        return _verbose;
      }

      void clear()
      {
        static_cast<DerivedClassifier*>(this)->_clearModel();
        _modelTrained=false;
      }

    protected:
      long _numberOfFeatures;
      long _numberOfClasses;
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
    };
  }
}
#endif //BASE_CLASSIFIER
