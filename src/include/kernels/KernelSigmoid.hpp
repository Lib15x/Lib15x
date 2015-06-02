#ifndef KERNEL_SIGMOID
#define KERNEL_SIGMOID
#include <core/Definitions.hpp>

namespace CPPLearn{
  namespace Kernels{

    class Sigmoid{
    public:
      Sigmoid(double gamma, double r) : _gamma{gamma}, _r{r} {}

      double operator()(const VectorXd& x, const VectorXd& y) const{
        if (x.size() != y.size()){
          throwException("inside dot kernel, vector size mismatch!");
        }
        return tanh(_gamma*x.dot(y)+_r);
      }

    private:
      double _gamma;
      double _r;
    };
  }
}
#endif //KERNEL_SIGMOID
