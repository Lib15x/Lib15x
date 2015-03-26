#include <mathematics/MatrixStatic.hpp>
#include <Eigen/Dense>

#include <gtest/gtest.h>
using namespace CPPLearn;
using namespace std;

TEST(MatrixStatic, Identity_check) {
  const unsigned numberOfRows=10;
  const unsigned numberOfCols=10;

  using MatrixType=MatrixStatic<double, numberOfRows, numberOfCols>;

  auto identityMatrix=MatrixType::Identity();
	for (size_t rowIndex=0; rowIndex<numberOfRows; ++rowIndex)
		for (size_t colIndex=0; colIndex<numberOfCols; ++colIndex)
			if (rowIndex == colIndex)
				EXPECT_EQ(identityMatrix(rowIndex, colIndex),1.0);
			else
				EXPECT_EQ(identityMatrix(rowIndex, colIndex),0.0);
}
