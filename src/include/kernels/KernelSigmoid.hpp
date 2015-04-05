#ifndef KERNEL_SIGMOID
#define KERNEL_SIGMOID
#include <core/Definitions.hpp>

namespace CPPLearn{
  namespace Kernels{

    class Sigmoid{
    public:
      Sigmoid(double gamma_, double r_) : gamma{gamma_}, r{r_} {}

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
}
#endif //KERNEL_SIGMOID
