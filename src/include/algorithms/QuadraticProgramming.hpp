#ifndef ALGORITHM_QUDRATIC_PROGRAMMING
#define ALGORITHM_QUDRATIC_PROGRAMMING

#include <core/Definitions.hpp>
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

namespace CPPLearn{
  namespace Algorithms{
    using namespace Ipopt;

    class QuadraticProgramming : public TNLP {
    public:
      /** constructor */
      QuadraticProgramming(const MatrixXd* Q_, const VectorXd* c_,
                           const MatrixXd* G_, const VectorXd* gL_, const VectorXd* gU_,
                           const VectorXd* xL_, const VectorXd* xU_,
                           const VectorXd* startPoint_)
        : Q{Q_}, c{c_}, G{G_}, gL{gL_}, gU{gU_}, xL{xL_}, xU{xU_},
        numberOfDimensions(c_->size()), numberOfAffineConstraints(gL->size()),
        solution{c_->size()}, startPoint{startPoint_} {};

      /** default destructor */
      virtual ~QuadraticProgramming(){}

      /**@name Overloaded from TNLP */
      //@{
      /** Method to return some info about the nlp */
      virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                Index& nnz_h_lag, IndexStyleEnum& index_style){
        n = numberOfDimensions;
        m = numberOfAffineConstraints;
        nnz_jac_g = n*m;
        nnz_h_lag = n*(n+1)/2;
        index_style = TNLP::C_STYLE;

        return true;
      }

      /** Method to return the bounds for my problem */
      virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                   Index m, Number* g_l, Number* g_u){
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);

        for (Index i=0; i<n; ++i) {
          x_l[i] = (*xL)(i);
          x_u[i] = (*xU)(i);
        }

        for (Index i=0; i<m; ++i) {
          g_l[i]= (*gL)(i);
          g_u[i]= (*gU)(i);
        }

        return true;
      }

      /** Method to return the starting point for the algorithm */
      virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                      bool init_z, Number* z_L, Number* z_U,
                                      Index m, bool init_lambda,
                                      Number* lambda){
        assert(init_x == true);
        assert(init_z == false);
        assert(init_lambda == false);
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);

        for (Index id=0; id<numberOfDimensions; ++id)
          x[id]=(*startPoint)(id);

        return true;
      }

      /** Method to return the objective value */
      virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value){
        assert(n == numberOfDimensions);
        Map<const VectorXd> x_map(x,n);
        obj_value = 0.5*x_map.dot((*Q)*x_map)+c->dot(x_map);

        return true;
      }

      /** Method to return the gradient of the objective */
      virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f){
        assert(n == numberOfDimensions);
        Map<const VectorXd> x_map(x,n);
        Map<VectorXd> grad_f_map(grad_f,n);
        grad_f_map=(*Q)*x_map+(*c);

        return true;
      }

      /** Method to return the constraint residuals */
      virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g){
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);

        Map<const VectorXd> x_map(x,n);
        Map<VectorXd> g_map(g,m);
        g_map=(*G)*x_map;

        return true;
      }

      /** Method to return:
       *   1) The structure of the jacobian (if "values" is NULL)
       *   2) The values of the jacobian (if "values" is not NULL)
       */
      virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                              Index m, Index nele_jac, Index* iRow, Index *jCol,
                              Number* values){
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        if (values == NULL) {
          Index entryIndex=0;
          for (Index rowIndex=0; rowIndex<m; ++rowIndex)
            for (Index colIndex=0; colIndex<n; ++colIndex){
              iRow[entryIndex] = rowIndex; jCol[entryIndex] = colIndex;
              ++entryIndex;
            }
        }
        else {
          Map<MatrixXd> value_map(values, n, m);
          value_map=(*G).transpose();
        }

        return true;
      }

      /** Method to return:
       *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
       *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
       */
      virtual bool eval_h(Index n, const Number* x, bool new_x,
                          Number obj_factor, Index m, const Number* lambda,
                          bool new_lambda, Index nele_hess, Index* iRow,
                          Index* jCol, Number* values){
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        if (values == NULL) {
          Index entryIndex=0;
          for (Index rowIndex=0; rowIndex<numberOfDimensions; ++rowIndex)
            for (Index colIndex=0; colIndex<=rowIndex; ++colIndex){
              iRow[entryIndex] = rowIndex; jCol[entryIndex] = colIndex;
              ++entryIndex;
            }
          assert(entryIndex == nele_hess);
        }
        else {
          Index id=0;
          for (Index rowId=0; rowId<n; ++rowId)
            for (Index colId=0; colId<=rowId; ++colId){
              values[id]=obj_factor*(*Q)(rowId, colId);
              ++id;
            }
          assert(id == nele_hess);
        }
        return true;
      }

      /** @name Solution Methods */
      //@{
      /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
      virtual void finalize_solution(SolverReturn status,
                                     Index n, const Number* x, const Number* z_L,
                                     const Number* z_U, Index m,
                                     const Number* g, const Number* lambda,
                                     Number obj_value, const IpoptData* ip_data,
                                     IpoptCalculatedQuantities* ip_cq){
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        for (Index entryIndex=0; entryIndex<n; ++entryIndex)
          solution(entryIndex)=x[entryIndex];
        solvedFlag=true;
        if (status != SUCCESS)
          cout<<"Warning: Quadratic solver did not find the optimal local minimum!"<<endl;;
      }

      const VectorXd& getOptimalSolution() const{
        if (!solvedFlag){
          throwException("Problem has not been solved yet! Please solve the problem first!");
        }
        return solution;
      }

    private:
      const MatrixXd* Q;
      const VectorXd* c;
      const MatrixXd* G;
      const VectorXd* gL;
      const VectorXd* gU;
      const VectorXd* xL;
      const VectorXd* xU;
      const Index numberOfDimensions;
      const Index numberOfAffineConstraints;
      VectorXd solution;
      bool solvedFlag=false;

      const VectorXd* startPoint;
    private:
      QuadraticProgramming(const QuadraticProgramming&);
      QuadraticProgramming& operator=(const QuadraticProgramming&);
    };

    VectorXd SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                       const MatrixXd& G, const VectorXd& gL, const VectorXd& gU,
                                       const VectorXd& xL, const VectorXd& xU,
                                       const VectorXd& startPoint, double tol){
      if (Q.cols() !=Q.rows()){
        throwException("Input matrix Q is not squares! "
                       "Number of rows: %ld; Number of cols: %ld.\n",
                       Q.rows(), Q.cols());
      }

      long int numberOfDimensions = Q.rows();

      if (startPoint.size() != numberOfDimensions){
        throwException("The dimension of the given initial point for iteration is wrong! "
                       "Dimension of the problem: %ld; Dimension of intial point provided: %ld.\n",
                       numberOfDimensions, startPoint.size());
      }

      if (c.size() != numberOfDimensions){
        throwException("Q and c have different dimensions ! "
                       "Dimension of Q: %ld; Dimension of c: %ld.\n",
                       numberOfDimensions, c.size());
      }

      if (G.cols() != numberOfDimensions){
        throwException("The dimension of constrain matrix G is incompatible "
                       "with the dimension of the problem! "
                       "Dimension of problem: %ld; Dimension of G: %ld.\n",
                       numberOfDimensions, G.cols());
      }

      if (G.rows() != gL.size() || G.rows() != gU.size()){
        throwException("Number of constraint mismatch! "
                       "Number of constraint from G: %ld; Number of constraint from gL: %ld;"
                       "Number of Constraint from gU: %ld.\n",
                       G.rows(), gL.size(), gU.size());
      }

      if (numberOfDimensions != xL.size() || numberOfDimensions != xU.size()){
        throwException("Constraint on variable is not compatible with problem dimension! "
                       "Problem dimension: %ld; Number of constraint from xL: %ld;"
                       "Number of Constraint from xU: %ld.\n",
                       numberOfDimensions, xL.size(), xU.size());
      }

      SmartPtr<QuadraticProgramming> qp = new QuadraticProgramming(&Q, &c,
                                                                   &G, &gL, &gU,
                                                                   &xL, &xU,
                                                                   &startPoint);

      SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
      app->RethrowNonIpoptException(true);

      app->Options()->SetNumericValue("tol", tol);
      app->Options()->SetStringValue("mu_strategy", "adaptive");
      //app->Options()->SetStringValue("derivative_test", "second-order");
      //app->Options()->SetStringValue("derivative_test_print_all", "yes");
      ApplicationReturnStatus status;
      status = app->Initialize();
      if (status != Solve_Succeeded) {
        throwException("Error during initializing the Quadratic Programming problem!");
      }

      status = app->OptimizeTNLP(qp);

      if (status != Solve_Succeeded) {
        cout<< "Warning: The QP solver FAILED!" << endl;
      }

      return qp->getOptimalSolution();
    }

    VectorXd SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                       const MatrixXd& G, const VectorXd& gL, const VectorXd& gU,
                                       const VectorXd& xL, const VectorXd& xU,
                                       const double tol){
      VectorXd startPoint=VectorXd::Random(c.size());
      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint, tol);
    }

    VectorXd SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                       const MatrixXd& G, const VectorXd& gL, const VectorXd& gU,
                                       const VectorXd& xL, const VectorXd& xU,
                                       const VectorXd& startPoint){
      double tol=1e-7;
      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint,tol);
    }

    VectorXd SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                       const MatrixXd& G, const VectorXd& gL, const VectorXd& gU,
                                       const VectorXd& xL, const VectorXd& xU){
      double tol=1e-7;
      VectorXd startPoint=VectorXd::Zero(c.size());
      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint,tol);
    }

  }
}

#endif //ALGORITHM_QUDRATIC_PROGRAMMING
