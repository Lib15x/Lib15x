#ifndef PREPROCESSING_STANDARD_SCALER
#define PREPROCESSING_STANDARD_SCALER

#include <core/Definitions.hpp>

namespace CPPLearn{
  namespace Preprocessing{

    /**
     * enter a brief description
     */
    class StandardScaler{
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
      void fit(const MatrixXd& data) {
        numberOfFeatures = data.cols();
        unsigned numberOfData = data.rows();
        if (numberOfData<=1){
          throwException("Error happened in standard scaler: "
                         "There is only (%u) data provided, which is smaller than 2.\n",
                         numberOfData);
        }

        colMean.resize(numberOfFeatures);
        colStd.resize(numberOfFeatures);
        for (size_t featIndex=0; featIndex<numberOfFeatures; ++featIndex){
          double mean=data.col(featIndex).mean();
          colMean(featIndex)=mean;
          VectorXd meanVec(numberOfData); meanVec.fill(mean);
          colStd(featIndex)=(data.col(featIndex)-meanVec).norm()/sqrt(numberOfData-1);
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

        unsigned numberOfData=data.rows();
        MatrixXd transformedData(data.rows(), data.cols());
        for (size_t dataIndex=0; dataIndex<numberOfData; ++dataIndex){
          for (size_t featIndex=0; featIndex<numberOfFeatures; ++featIndex){
            transformedData(dataIndex,featIndex)=
              (data(dataIndex, featIndex)-colMean(featIndex))/colStd(featIndex);
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
      VectorXd colMean;
      VectorXd colStd;
    };
  }
}

#endif //PREPROCESSING_STANDARD_SCALER
