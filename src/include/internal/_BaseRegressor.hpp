#ifndef BASE_REGRESSOR
#define BASE_REGRESSOR

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template<class DerivedRegressor>
    class _BaseRegressor {
    public:
      static const ProblemType ModelType = ProblemType::Regression;
      static constexpr const char* ModelName=DerivedRegressor::ModelName;
      static double
      LossFunction(const Labels& predictedLabels, const Labels& testLabels)
      {
        if (predictedLabels._labelType != ProblemType::Regression ||
            testLabels._labelType != ProblemType::Regression){
          throwException("Error happen when computing regression loss: "
                         "Input labelType must be Regression!\n");
        }

        const VectorXd& predictedLabelData=predictedLabels._labelData;
        const VectorXd& testLabelData=testLabels._labelData;

        if (predictedLabelData.size() != testLabelData.size()){
          throwException("Error happen when computing regression loss: "
                         "The inpute two labels have different sizes. "
                         "sizes of the predicted labels: (%ld); "
                         "sizes of the test labels: (%ld).\n",
                         predictedLabelData.size(), testLabelData.size());
        }

        const long numberOfData = testLabelData.size();
        double loss=0.0;
        for (long dataId = 0; dataId<numberOfData; ++dataId)
          if (predictedLabelData(dataId) != std::numeric_limits<double>::max() &&
              testLabelData(dataId) != std::numeric_limits<double>::max()) {
            double diff = predictedLabelData(dataId) - testLabelData(dataId);
            loss+=diff*diff;
          }

        return loss;
      }

      explicit _BaseRegressor(const long numberOfFeatures) :
        _numberOfFeatures{numberOfFeatures} { }

      void
      train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ModelType) {
          throwException("Error happen when training %s model: "
                         "Input labelType must be Regression!\n", ModelName);
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

        VectorXd balancedWeights(trainData.rows()); balancedWeights.fill(1.0);
        static_cast<DerivedRegressor*>(this)->train(trainData, trainLabels, balancedWeights);
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
        Labels predictedLabels{ProblemType::Regression};
        predictedLabels._labelData.resize(testData.rows());
        predictedLabels._labelData.fill(std::numeric_limits<double>::max());

        for (auto testDataId : testIndices){
          Map<const VectorXd> instance(&testData(testDataId, 0), _numberOfFeatures);
          predictedLabels._labelData(testDataId) =
            static_cast<const DerivedRegressor*>(this)->predictOne(instance);
        }

        return predictedLabels;
      }

      const long&
      getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }

      VerboseFlag&
      whetherVerbose()
      {
        return _verbose;
      }

      void clear()
      {
        static_cast<DerivedRegressor*>(this)->_clearModel();
        _modelTrained=false;
      }

    protected:
      long _numberOfFeatures;
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
    };
  }
}
#endif //BASE_REGRESSOR
