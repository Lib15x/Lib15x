#ifndef PREPROCESSING_MINMAX_SCALER
#define PREPROCESSING_MINMAX_SCALER

#include <core/Definitions.hpp>

namespace CPPLearn
{
  namespace Preprocessing
  {
    /**
     * enter a brief description
     */
    class MinMaxScaler
    {
    public:
      /**
       * add brief description
       *
       * @param
       */
      MinMaxScaler(const double lowerBound=-1.0, const double upperBound=1.0):
        _lowerBound{lowerBound}, _upperBound{upperBound} {}
      /**
       * brief description
       *
       * @param data describe parameter
       */
      void fit(const MatrixXd& data)
      {
        _numberOfFeatures = data.cols();
        _colMax.resize(_numberOfFeatures);
        _colMin.resize(_numberOfFeatures);
        for (long featIndex=0; featIndex<_numberOfFeatures; ++featIndex){
          _colMax(featIndex)=data.col(featIndex).maxCoeff();
          _colMin(featIndex)=data.col(featIndex).minCoeff();
          if (_colMax(featIndex)==_colMin(featIndex))
            printf("Warning from MinMax scaler: "
                   "I found the data for feature No.(%ld) are the same.", featIndex);
        }
        _scalerFitted=true;
      }

      /**
       * brief description
       *
       * @param data describe parameter
       */
      MatrixXd transform(const MatrixXd& data) const
      {
        if (!_scalerFitted){
          throwException("Error happened when transform data using MinMax scaler: "
                         "Scaler has not been fitted yet!");
        }

        if (data.cols() != _numberOfFeatures){
          throwException("Error happened when transform data using MinMax scaler: "
                         "expecting number of features from scaler: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         _numberOfFeatures, data.cols());
        }

        double range = _upperBound-_lowerBound;
        long numberOfData = data.rows();
        MatrixXd transformedData(data.rows(), data.cols());
        for (long dataIndex=0; dataIndex<numberOfData; ++dataIndex){
          for (long featIndex=0; featIndex<_numberOfFeatures; ++featIndex){
            if (_colMax(featIndex) == _colMin(featIndex))
              transformedData(dataIndex,featIndex)=data(dataIndex,featIndex);
            else
              transformedData(dataIndex,featIndex)=
                _lowerBound+range/(_colMax(featIndex)-_colMin(featIndex))
                *(data(dataIndex, featIndex)-_colMin(featIndex));
          }
        }
        return transformedData;
      }

      /**
       * brief description
       *
       * @param data describe parameter
       */
      MatrixXd fitTransform(const MatrixXd& data)
      {
        fit(data);
        return transform(data);
      }

      /**
       * Clear the model.
       */
      void clear()
      {
        _numberOfFeatures=0;
        _scalerFitted=false;
      }

      VerboseFlag& setVerbose(){
        return _verbose;
      }

    private:
      long _numberOfFeatures;
      bool _scalerFitted=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
      double _lowerBound;
      double _upperBound;
      VectorXd _colMax;
      VectorXd _colMin;
    };
  }
}

#endif //PREPROCESSING_MINMAX_SCALER
