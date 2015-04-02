#include <core/Definitions.hpp>

namespace CPPLearn{
	class KernelDot{
	public:
		KernelDot(){}

		double operator()(const VectorXd& x, const VectorXd& y) const {
      if (x.size() != y.size())
        throw std::runtime_error("inside dot kernel, vector size mismatch!");
			return x.dot(y);
		}
	};
}
