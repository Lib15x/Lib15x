#ifndef PREPROCESSING_STANDARD_SCALER
#define PREPROCESSING_STANDARD_SCALER

#include "../core/Definitions.hpp"

namespace CPPLearn
{
  namespace Preprocessing
  {
    /**
     * enter a brief description
     */
    class StandardScaler
    {
    public:
      /**
       * Creates the model, with empty model initialized.
       *
       * @param
       */
      StandardScaler() {}

      /**
       * brief description
       *
       * @param data describe parameter
       */
      void fit(const MatrixXd& data)
      {
        _numberOfFeatures = data.cols();
        long numberOfData = data.rows();
        if (numberOfData<=1){
          throwException("Error happened in standard scaler: "
                         "There is only (%ld) data provided, which is smaller than 2.\n",
                         numberOfData);
        }

        _colMean.resize(_numberOfFeatures);
        _colStd.resize(_numberOfFeatures);
        for (long featIndex=0; featIndex<_numberOfFeatures; ++featIndex){
          double mean=data.col(featIndex).mean();
          _colMean(featIndex)=mean;
          double var=0;
          for (long dataIndex=0; dataIndex<numberOfData; ++dataIndex)
            var+=(data(dataIndex,featIndex)-mean)*(data(dataIndex,featIndex)-mean);
          if (var==0.0)
            printf("Warning from MinMax scaler: "
                   "I found the data for feature No.(%ld) are the same.", featIndex);
          var/=static_cast<double>(numberOfData)-1.0;
          _colStd(featIndex)=sqrt(var);
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

        long numberOfData=data.rows();
        MatrixXd transformedData(data.rows(), data.cols());
        for (long dataIndex=0; dataIndex < numberOfData; ++dataIndex){
          for (long featIndex=0; featIndex < _numberOfFeatures; ++featIndex){
            if (_colStd(featIndex) == 0.0)
              transformedData(dataIndex,featIndex)=0.0;
            else
              transformedData(dataIndex,featIndex)=
                (data(dataIndex, featIndex)-_colMean(featIndex))/_colStd(featIndex);
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

      VerboseFlag& setVerbose()
      {
        return _verbose;
      }

    private:
      long _numberOfFeatures;
      bool _scalerFitted=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
      VectorXd _colMean;
      VectorXd _colStd;
    };
  }
}

#endif //PREPROCESSING_STANDARD_SCALER
