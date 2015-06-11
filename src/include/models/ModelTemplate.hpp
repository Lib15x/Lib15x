#ifndef MODEL_CLASSIFICATION_TREE
#define MODEL_CLASSIFICATION_TREE

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"

namespace CPPLearn
{
  namespace Models
  {
    class ClassificationTree {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="ClassificationTree";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::classificationZeroOneLossFunction;

      ClassificationTree(const long numberOfFeatures) :
        _numberOfFeatures{numberOfFeatures} { }

      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Classification){
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

        _modelTrained=true;
      }

      Labels predict(const MatrixXd& testData) const
      {
        if (!_modelTrained){
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

        long numberOfTests=testData.rows();
        Labels predictedLabels{ProblemType::Classification};
        predictedLabels._labelData.resize(numberOfTests);

        for (long rowIndex=0; rowIndex<numberOfTests; ++rowIndex){
          Map<const VectorXd> instance(&testData(rowIndex, 0), _numberOfFeatures);
          predictedLabels._labelData(rowIndex) = predictOne(instance);
        }

        return predictedLabels;
      }

      double predictOne(const VectorXd& instance) const
      {
        //need to take care of bias term
        double result =0;

        return result;
      }

      const long& getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }


      /**
       * Clear the model.
       */
      void clear()
      {
        _modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return _verbose;
      }


    private:
      const long _numberOfFeatures;
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;

    };
  }
}

#endif //MODEL_CLASSIFICATION_TREE
