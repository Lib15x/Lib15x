#ifndef PREPROCESSING_MINMAX_SCALER
#define PREPROCESSING_MINMAX_SCALER

#include <core/Definitions.hpp>

namespace CPPLearn{
  namespace Preprocessing{

    /**
     * enter a brief description
     */
    class MinMaxScaler{
    public:

      /**
       * add brief description
       *
       * @param
       */
      MinMaxScaler(const double lowerBound_=-1.0, const double upperBound_=1.0):
        lowerBound{lowerBound_}, upperBound{upperBound_} {}
      /**
       * brief description
       *
       * @param data describe parameter
       */
      void fit(const MatrixXd& data) {
        numberOfFeatures = data.cols();
        colMax.resize(numberOfFeatures);
        colMin.resize(numberOfFeatures);
        for (size_t featIndex=0; featIndex<numberOfFeatures; ++featIndex){
          colMax(featIndex)=data.col(featIndex).maxCoeff();
          colMin(featIndex)=data.col(featIndex).minCoeff();
          if (colMax(featIndex)==colMin(featIndex))
            printf("Warning from MinMax scaler: "
                   "I found the data for feature No.(%lu) are the same.", featIndex);
        }
        scalerFitted=true;
      }

      /**
       * brief description
       *
       * @param data describe parameter
       */
      MatrixXd transform(const MatrixXd& data) const {
        if (!scalerFitted){
          throwException("Error happened when transform data using MinMax scaler: "
                         "Scaler has not been fitted yet!");
        }

        if (data.cols() != numberOfFeatures){
          throwException("Error happened when transform data using MinMax scaler: "
                         "expecting number of features from scaler: (%u); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, data.cols());
        }

        double range=upperBound-lowerBound;
        unsigned numberOfData=data.rows();
        MatrixXd transformedData(data.rows(), data.cols());
        for (size_t dataIndex=0; dataIndex<numberOfData; ++dataIndex){
          for (size_t featIndex=0; featIndex<numberOfFeatures; ++featIndex){
            if (colMax(featIndex)==colMin(featIndex))
              transformedData(dataIndex,featIndex)=0.0;
            else
            transformedData(dataIndex,featIndex)=
              lowerBound+range/(colMax(featIndex)-colMin(featIndex))
              *(data(dataIndex, featIndex)-colMin(featIndex));
          }
        }
        return transformedData;
      }

      /**
       * brief description
       *
       * @param data describe parameter
       */
      MatrixXd fitTransform(const MatrixXd& data) {
        fit(data);
        return transform(data);
      }

      /**
       * Clear the model.
       */
      void clear(){
        numberOfFeatures=0;
        scalerFitted=false;
      }

      VerboseFlag& setVerbose(){
        return verbose;
      }

    private:
      unsigned numberOfFeatures;
      bool scalerFitted=false;
      VerboseFlag verbose = VerboseFlag::Quiet;
      double lowerBound;
      double upperBound;
      VectorXd colMax;
      VectorXd colMin;
    };
  }
}

#endif //PREPROCESSING_MINMAX_SCALER
