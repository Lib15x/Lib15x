#include <core/Definitions.hpp>

namespace CPPLearn{
  class KernelSigmoid{
  public:
    KernelSigmoid(double gamma_, double r_) : gamma{gamma_}, r{r_} {}

    double operator()(const VectorXd& x, const VectorXd& y) const{
      if (x.size() != y.size())
        throw std::runtime_error("inside dot kernel, vector size mismatch!");
      return tanh(gamma*x.dot(y)+r);
    }
  private:
    double gamma;
    double r;
  };
}
