#include <iostream>
#include <Eigen/Dense>
using namespace std;

namespace CPPLearn{

  template <typename Type, unsigned Rows, unsigned Cols>
  class MatrixStatic{

  public:
    static MatrixStatic<Type, Rows, Cols> Identity() {
      MatrixStatic<Type, Rows, Cols> identity;
      identity.eigenMatrix=Eigen::Matrix<Type, Rows, Cols>::Identity();
      return identity;
    }

    MatrixStatic() {}

    size_t cols() const {
      return eigenMatrix.cols();
    }

    size_t rows() const {
      return eigenMatrix.rows();
    }

    const Type& operator()(size_t rowIndex, size_t colIndex) const {
      return eigenMatrix(rowIndex, colIndex);
    }

    Type& operator()(size_t rowIndex, size_t colIndex){
      return eigenMatrix(rowIndex, colIndex);
    }

    Type determinant(){
      return eigenMatrix.determinant();
    }

    MatrixStatic<Type, Rows, Cols>
    inverse(){
      MatrixStatic<Type, Rows, Cols> inv;
      inv.eigenMatrix=eigenMatrix.inverse();
      return inv;
    }

    void display() const {
      cout<<eigenMatrix<<endl;
    }

    void fill(Type value){
      eigenMatrix.fill(value);
    }


  private:
    Eigen::Matrix<Type, Rows, Cols> eigenMatrix;
  };

}
