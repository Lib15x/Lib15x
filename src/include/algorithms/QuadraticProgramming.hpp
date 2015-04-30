#ifndef ALGORITHM_QUDRATIC_PROGRAMMING
#define ALGORITHM_QUDRATIC_PROGRAMMING

#include <core/Definitions.hpp>
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

namespace CPPLearn
{
  namespace Algorithms
  {
    struct OptSolution
    {
      double objOptimal=0;
      VectorXd minimizer;
      bool solverSucess=false;
    };

    class QuadraticProgramming : public Ipopt::TNLP
    {
    public:
      /** constructor */
      QuadraticProgramming(const MatrixXd* Q_, const VectorXd* c_,
                           const MatrixXd* G_, const VectorXd* gL_, const VectorXd* gU_,
                           const VectorXd* xL_, const VectorXd* xU_,
                           const VectorXd* startPoint_)
        : Q{Q_}, c{c_}, G{G_}, gL{gL_}, gU{gU_}, xL{xL_}, xU{xU_},
        numberOfDimensions(c_->size()), numberOfAffineConstraints(gL->size()),
        startPoint{startPoint_} {};

      /** default destructor */
      virtual ~QuadraticProgramming(){}

      /**@name Overloaded from TNLP */
      //@{
      /** Method to return some info about the nlp */
      virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                                Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
      {
        n = numberOfDimensions;
        m = numberOfAffineConstraints;
        nnz_jac_g = n*m;
        nnz_h_lag = n*(n+1)/2;
        index_style = TNLP::C_STYLE;

        return true;
      }

      /** Method to return the bounds for my problem */
      virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                   Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
      {
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);

        for (Ipopt::Index i=0; i<n; ++i) {
          x_l[i] = (*xL)(i);
          x_u[i] = (*xU)(i);
        }

        for (Ipopt::Index i=0; i<m; ++i) {
          g_l[i]= (*gL)(i);
          g_u[i]= (*gU)(i);
        }

        return true;
      }

      /** Method to return the starting point for the algorithm */
      virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                      bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                                      Ipopt::Index m, bool init_lambda,
                                      Ipopt::Number* lambda)
      {
        assert(init_x == true);
        assert(init_z == false);
        assert(init_lambda == false);
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);

        for (Ipopt::Index id=0; id<numberOfDimensions; ++id)
          x[id]=(*startPoint)(id);

        return true;
      }

      /** Method to return the objective value */
      virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                          Ipopt::Number& obj_value)
      {
        assert(n == numberOfDimensions);
        Map<const VectorXd> x_map(x,n);
        obj_value = 0.5*x_map.dot((*Q)*x_map)+c->dot(x_map);

        return true;
      }

      /** Method to return the gradient of the objective */
      virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                               Ipopt::Number* grad_f)
      {
        assert(n == numberOfDimensions);
        Map<const VectorXd> x_map(x,n);
        Map<VectorXd> grad_f_map(grad_f,n);
        grad_f_map=(*Q)*x_map+(*c);

        return true;
      }

      /** Method to return the constraint residuals */
      virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                          Ipopt::Index m, Ipopt::Number* g)
      {
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
      virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                              Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow,
                              Ipopt::Index *jCol, Ipopt::Number* values)
      {
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        if (values == NULL) {
          Ipopt::Index entryIndex=0;
          for (Ipopt::Index rowIndex=0; rowIndex<m; ++rowIndex)
            for (Ipopt::Index colIndex=0; colIndex<n; ++colIndex){
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
      virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                          Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
                          bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                          Ipopt::Index* jCol, Ipopt::Number* values)
      {
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        if (values == NULL) {
          Ipopt::Index entryIndex=0;
          for (Ipopt::Index rowIndex=0; rowIndex<numberOfDimensions; ++rowIndex)
            for (Ipopt::Index colIndex=0; colIndex<=rowIndex; ++colIndex){
              iRow[entryIndex] = rowIndex; jCol[entryIndex] = colIndex;
              ++entryIndex;
            }
          assert(entryIndex == nele_hess);
        }
        else {
          Ipopt::Index id=0;
          for (Ipopt::Index rowId=0; rowId<n; ++rowId)
            for (Ipopt::Index colId=0; colId<=rowId; ++colId){
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
      virtual void finalize_solution(Ipopt::SolverReturn status,
                                     Ipopt::Index n,
                                     const Ipopt::Number* x,
                                     const Ipopt::Number* z_L,
                                     const Ipopt::Number* z_U,
                                     Ipopt::Index m,
                                     const Ipopt::Number* g,
                                     const Ipopt::Number* lambda,
                                     Ipopt::Number obj_value,
                                     const Ipopt::IpoptData* ip_data,
                                     Ipopt::IpoptCalculatedQuantities* ip_cq)
      {
        assert(n == numberOfDimensions);
        assert(m == numberOfAffineConstraints);
        solvedFlag=true;

        solution.minimizer.resize(numberOfDimensions);
        for (Ipopt::Index entryIndex=0; entryIndex<n; ++entryIndex)
          solution.minimizer(entryIndex)=x[entryIndex];

        solution.objOptimal=obj_value;
        solution.solverSucess = status == Ipopt::SUCCESS ? true: false;

      }

      const OptSolution& getOptimalSolution() const
      {
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
      const Ipopt::Index numberOfDimensions;
      const Ipopt::Index numberOfAffineConstraints;
      bool solvedFlag=false;
      const VectorXd* startPoint;
      OptSolution solution;

    private:
      QuadraticProgramming(const QuadraticProgramming&);
      QuadraticProgramming& operator=(const QuadraticProgramming&);
    };


    OptSolution SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                          const MatrixXd& G, const VectorXd& gL, const VectorXd& gU,
                                          const VectorXd& xL, const VectorXd& xU,
                                          const VectorXd& startPoint, double tol,
                                          const bool ipoptVerboseFlag=false)
    {
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

      Ipopt::SmartPtr<QuadraticProgramming> qp =
        new QuadraticProgramming(&Q, &c, &G, &gL, &gU, &xL, &xU, &startPoint);

      //disable ipopt output
      Ipopt::SmartPtr<Ipopt::IpoptApplication> app =
        new Ipopt::IpoptApplication (ipoptVerboseFlag);

      app->RethrowNonIpoptException(true);

      app->Options()->SetNumericValue("tol", tol);
      app->Options()->SetStringValue("mu_strategy", "adaptive");
      app->Options()->SetStringValue("hessian_approximation", "limited-memory");
      //app->Options()->SetIntegerValue("print_level", 5);
      //app->Options()->SetStringValue("derivative_test", "second-order");
      //app->Options()->SetStringValue("derivative_test_print_all", "yes");
      Ipopt::ApplicationReturnStatus status;
      status = app->Initialize();
      if (status != Ipopt::Solve_Succeeded) {
        throwException("Error during initializing the Quadratic Programming problem!");
      }

      status = app->OptimizeTNLP(qp);

      OptSolution solution=qp->getOptimalSolution();

      if (!solution.solverSucess) {
        cout<< "Warning: The Quadratic Programming solver didnot find a local minimum! " << endl;
      }

      return solution;
    }

    OptSolution SolveQudraticProgramming(const MatrixXd& Q, const VectorXd& c,
                                         const MatrixXd& G,
                                         const VectorXd& gL, const VectorXd& gU,
                                         const VectorXd& xL, const VectorXd& xU,
                                         const double tol,
                                         const bool ipoptVerboseFlag=false)
    {
      VectorXd startPoint=VectorXd::Random(c.size());
      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint, tol, ipoptVerboseFlag);
    }

    OptSolution SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                          const MatrixXd& G, const VectorXd& gL,
                                          const VectorXd& gU,
                                          const VectorXd& xL, const VectorXd& xU,
                                          const VectorXd& startPoint,
                                          const bool ipoptVerboseFlag=false)
    {
      double tol=1e-7;
      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint,tol, ipoptVerboseFlag);
    }

    OptSolution SolveQudraticProgramming (const MatrixXd& Q, const VectorXd& c,
                                          const MatrixXd& G,
                                          const VectorXd& gL, const VectorXd& gU,
                                          const VectorXd& xL, const VectorXd& xU,
                                          const bool ipoptVerboseFlag=false)
    {
      double tol=1e-7;
      VectorXd startPoint=VectorXd::Zero(c.size());

      return SolveQudraticProgramming(Q, c, G, gL, gU, xL, xU, startPoint,tol, ipoptVerboseFlag);
    }

  }
}

#endif //ALGORITHM_QUDRATIC_PROGRAMMING
