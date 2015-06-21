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
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::regressionSquaredNormLossFunction;

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

        vector<long> indicesForAllTrainData(trainData.rows());
        std::iota(std::begin(indicesForAllTrainData), std::end(indicesForAllTrainData), 0);
        static_cast<DerivedRegressor*>(this)->train(trainData, trainLabels,
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
        Labels predictedLabels{ProblemType::Regression};
        predictedLabels._labelData.resize(testData.rows());
        predictedLabels._labelData.fill(-1.0);

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
