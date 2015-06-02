#ifndef KERNEL_RBF
#define KERNEL_RBF

#include <core/Definitions.hpp>

namespace CPPLearn{
  namespace Kernels{

    class RBF{
    public:
      explicit RBF(double gamma) : _gamma(gamma) {
        if (_gamma<=0){
          throwException("inconstructor of RBF kernel, "
                         "gamma should greater than zero!");
        }
      }

      double operator()(const VectorXd& x, const VectorXd& y) const {
        if (x.size() != y.size()){
          throwException("inside RBF kernel, vector size mismatch!");
        }
        return exp(-_gamma*(x-y).squaredNorm());
      }

    private:
      double _gamma;
    };
  }
}

#endif //KERNEL_RBF
