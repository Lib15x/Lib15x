#include <core/Definitions.hpp>
#include <core/Utilities.hpp>
#include <algorithms/QuadraticProgramming.hpp>

using namespace CPPLearn;

int main(int argc, char* argv[]){
  ignoreUnusedVariables(argc, argv);
  MatrixXd data(9,2);
  data<<1,1,
    1, 3,
    2, 2,
    2, 4,
    3, 2,
    3, 4,
    4, 1,
    4, 3,
    5, 5;
  VectorXd labels(9);
  labels<<1, 1, 1, 1, 1, -1, -1, -1, -1;
  MatrixXd hessian(9,9);
  for (size_t rowId=0; rowId<9; ++rowId)
    for (size_t colId=0; colId<9; ++colId)
      hessian(rowId, colId)=labels(rowId)*labels(colId)*data.row(rowId).dot(data.row(colId));

  VectorXd drift(9); drift.fill(-1);
  VectorXd gL(1); gL.fill(0);
  VectorXd gU(1); gU.fill(0);
  VectorXd xL(9); xL.fill(0);
  VectorXd xU(9); xU.fill(1e9);
  MatrixXd G=labels.transpose();

  VectorXd solution=Algorithms::SolveQudraticProgramming (hessian, drift,
                                                          G, gL, gU,
                                                          xL, xU);


  ////cout<<0.5*solution.transpose()*hessian*solution+drift.dot(solution)<<endl;
  cout<<solution<<endl;
  ////MatrixXd AA(3,3);
  ////for (size_t i=0; i<9;++i)
  ////AA(i)=i;
  ////
  ////Map<VectorXd> V(&AA(0), 9);
  ////cout<<V<<endl;


  return 0;
}
