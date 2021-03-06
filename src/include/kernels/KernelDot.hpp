#ifndef KERNEL_DOT
#define KERNEL_DOT

#include "../core/Definitions.hpp"

namespace Lib15x
{
  namespace Kernels
  {
    class Dot{
    public:
      Dot(){}

      double operator()(const VectorXd& x, const VectorXd& y) const {
        if (x.size() != y.size()){
          throwException("inside dot kernel, vector size mismatch!");
        }
        return x.dot(y);
      }
    };
  }
}

#endif //KERNEL_DOT
