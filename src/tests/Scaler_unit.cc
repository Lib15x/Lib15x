#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <preprocessing/StandardScaler.hpp>
#include <preprocessing/MinMaxScaler.hpp>
#include <gtest/gtest.h>

using namespace Lib15x;

TEST(Scaler, Standard_test)
{
  using Scaler=Preprocessing::StandardScaler;

  long numberOfFeatures=10;
  long numberOfData=100;

  MatrixXd data=MatrixXd::Random(numberOfData,numberOfFeatures);

  Scaler scaler;
  MatrixXd transformedData=scaler.fitTransform(data);

  for (long featIndex=0; featIndex<numberOfFeatures; ++featIndex){
    double mean=transformedData.col(featIndex).mean();
    double stdDev=Utilities::computeStandardDeviation(transformedData.col(featIndex));
    EXPECT_NEAR(mean, 0.0, 1e-10);
    EXPECT_NEAR(stdDev, 1.0, 1e-10);
  }
}
