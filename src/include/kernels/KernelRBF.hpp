#include <core/Definitions.hpp>

namespace CPPLearn{
  class KernelRBF{
  public:
    KernelRBF(double gamma_) : gamma(gamma_) {
      if (gamma<=0)
				throw std::runtime_error("inconstructor of RBF kernel, gamma should greater than zero!");
    }

    double operator()(const VectorXd& x, const VectorXd& y) const {
      if (x.size() != y.size())
        throw std::runtime_error("inside dot kernel, vector size mismatch!");
      return exp(-gamma*(x-y).squaredNorm());
    }
  private:
    double gamma;
  };
}
