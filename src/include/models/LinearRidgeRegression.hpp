#include <core/Definitions.hpp>

namespace CPPLearn{
  class LinearRidgeRegression{
  public:
    LinearRidgeRegression(const double regularizer_) : regularizer{regularizer_} {}

    void train(const MatrixXd& trainData, const VectorXd& trainLabel) {
      if (trainData.rows() != trainLabel.size())
        throw std::runtime_error("data and label size mismatch!");

      size_t numberOfFeatures = trainData.cols();
      auto matrixC = trainData.transpose()*trainData +
        regularizer*MatrixXd::Identity(numberOfFeatures, numberOfFeatures);
      parameters=matrixC.llt().solve(trainData.transpose()*trainLabel);
      modelTrained = true;
    }

    VectorXd predict(const MatrixXd& testData){
      if (!modelTrained)
        throw std::runtime_error("model has not been trained yet");

      if (testData.cols() != parameters.size())
        throw std::runtime_error("data and label size mismatch!");

      VectorXd predictedLabel = testData*parameters;
      return predictedLabel;
    }

  private:
    VectorXd parameters;
    double regularizer;
    bool modelTrained = false;
  };
}
