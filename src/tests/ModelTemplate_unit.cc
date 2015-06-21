#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <models/RegressorTemplate.hpp>
#include <models/ClassifierTemplate.hpp>
#include <gtest/gtest.h>

using namespace CPPLearn;

TEST(ModelTemplate, Classifier_test)
{
  long numberOfFeatures = 3;
  long numberOfData = 10;
  long numberOfClasses = 2;
  MatrixXd data = MatrixXd::Random(numberOfData, numberOfFeatures);
  Labels labels{ProblemType::Classification};
  labels._labelData.resize(numberOfData);
  for (long dataId=0; dataId<numberOfData; ++dataId)
    labels._labelData(dataId)=rand() % 2;

  Models::ClassifierTemplate model{numberOfFeatures, numberOfClasses};
  model.train(data, labels);
  Labels predictedLabel = model.predict(data);

  for (long testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(predictedLabel._labelData[testIndex], 0);

  model.clear();
}

TEST(ModelTemplate, Regression_test)
{
  long numberOfFeatures = 3;
  long numberOfData = 10;
  MatrixXd data = MatrixXd::Random(numberOfData, numberOfFeatures);
  Labels labels{ProblemType::Regression};
  labels._labelData.resize(numberOfData);
  for (long dataId=0; dataId<numberOfData; ++dataId)
    labels._labelData(dataId)=(double)rand() / RAND_MAX;

  Models::RegressorTemplate model{numberOfFeatures};
  model.train(data, labels);
  Labels predictedLabel = model.predict(data);

  for (long testIndex=0; testIndex<numberOfData; ++testIndex)
    EXPECT_EQ(predictedLabel._labelData[testIndex], 0);
  model.clear();
}
