#include <core/Definitions.hpp>

namespace CPPLearn{
  class KernelPolynomial{
  public:
    KernelPolynomial(double gamma_, double r_, unsigned d_) :
      gamma{gamma_}, r{r_}, d{d_} {
        if (gamma <= 0 || d == 0)
          throw std::runtime_error("in constructor of polynomial kernel, "
                                   "gamma and d should greater than zero!");
      }

    double operator()(const VectorXd& x, const VectorXd& y) const {
      if (x.size() != y.size())
        throw std::runtime_error("inside dot kernel, vector size mismatch!");
      double result=1;
      double base=(gamma*x.dot(y)+r);
      for (size_t i=0; i<d; ++i)
        result *= base;
      return result;
    }

  private:
    double gamma;
    double r;
    unsigned d;
  };
}
