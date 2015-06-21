#ifndef KERNEL_POLYNOMIAL
#define KERNEL_POLYNOMIAL

#include "../core/Definitions.hpp"

namespace CPPLearn{
  namespace Kernels{

    class Polynomial{
    public:
      Polynomial(long d, double gamma, double r) :
        _d{d}, _gamma{gamma}, _r{r} {
          if (_gamma <= 0 || _d == 0){
            throwException("in constructor of polynomial kernel, "
                           "gamma and d should greater than zero!");
          }
        }

      double operator()(const VectorXd& x, const VectorXd& y) const {
        if (x.size() != y.size()){
          throwException("inside dot kernel, vector size mismatch!");
        }
        double result=1;
        double base=(_gamma*x.dot(y)+_r);
        for (long i=0; i<_d; ++i)
          result *= base;
        return result;
      }

    private:
      long _d;
      double _gamma;
      double _r;
    };
  }
}
#endif //KERNEL_POLYNOMIAL
