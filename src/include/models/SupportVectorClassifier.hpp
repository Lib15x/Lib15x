#ifndef MODEL_SUPPORT_VECTOR_CASSIFIER
#define MODEL_SUPPORT_VECTOR_CASSIFIER

#include <core/Definitions.hpp>
#include <algorithms/QuadraticProgramming.hpp>

namespace CPPLearn
{
  namespace Models
  {
    /**
     * A Support Vector Classifier based on libsvm implementation.
     */
    template<class Kernel>
    class SupportVectorClassifier
    {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="SupportVectorClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        &Utilities::classificationZeroOneLossFunction;

      /**
       * Creates the model, with empty model initialized.
       *
       * @param kernel_ kernel function object used for transformation, typical
       * types include linear, RBF or sigmoid.
       * @param numberOfFeatures_ number of features of the model,
       * required user provided befor hand.
       * @param C_ regularization constant.
       * @param tol_ stopping critiria.
       */
      SupportVectorClassifier(Kernel kernel_, size_t numberOfFeatures_,
                              double C_=1.0, double tol_=1e-7) :
        kernel{kernel_}, numberOfFeatures{numberOfFeatures_}, C{C_}, tol{tol_} { }

      /**
       * Train the model, the provided data and labels should have the some number of instance.
       *
       * @param trainData predictors, the number of columns should be the
       * same as number of features.
       * @param trainLabels contains the labels used for training.
       */
      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels.labelType != ProblemType::Classification){
          throwException("Error happen when training LibSVM model: "
                         "Input labelType must be Classification!\n");
        }
        const VectorXd& labelData=trainLabels.labelData;

        if ((unsigned)trainData.cols() != numberOfFeatures){
          throwException("Error happen when training model, invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }

        const size_t numberOfTrainData = trainData.rows();
        VectorXd localLabels(numberOfTrainData);
        for (size_t labelIndex=0; labelIndex<numberOfTrainData; ++ labelIndex){
          if (labelData(labelIndex) != 0 && labelData(labelIndex) != 1){
            throwException("In SV classifier: the provided labels should only contain 0 and 1!");
          }
          localLabels[labelIndex]= labelData[labelIndex] == 0 ? -1: 1;
        }

        MatrixXd hessian(numberOfTrainData, numberOfTrainData);

        for (size_t rowIndex=0; rowIndex<numberOfTrainData; ++rowIndex){
          Map<const VectorXd> dataI(&trainData(rowIndex,0), numberOfFeatures);
          for (size_t colIndex=0; colIndex<numberOfTrainData; ++colIndex){
            Map<const VectorXd> dataJ(&trainData(colIndex,0), numberOfFeatures);
            hessian(rowIndex, colIndex)= localLabels(rowIndex)*localLabels(colIndex)*
              kernel(dataI, dataJ);
          }
        }

        VectorXd c(numberOfTrainData); c.fill(-1);
        VectorXd gL(1); gL.fill(0);
        VectorXd gU(1); gU.fill(0);
        VectorXd xL(numberOfTrainData); xL.fill(0);
        VectorXd xU(numberOfTrainData); xU.fill(C);
        MatrixXd G=localLabels.transpose();

        Algorithms::OptSolution solution=
          Algorithms::SolveQudraticProgramming (hessian, c, G, gL, gU, xL, xU, tol, moreOptInfo);

        size_t anySupportVecIndex=0;
        VectorXd& alpha=solution.minimizer;
        for (size_t index=0; index<numberOfTrainData; ++ index){
          assert(alpha(index) >= 0.0);
          if (fabs(alpha(index)) > small){
            VectorXd localVec=trainData.row(index);
            double coeff=alpha(index)*localLabels(index);;
            supportVectors.push_back(std::pair<double,VectorXd>(coeff, std::move(localVec)));
            anySupportVecIndex=index;
          }
        }

        b=localLabels(anySupportVecIndex);
        VectorXd x_m=trainData.row(anySupportVecIndex);
        for (auto svIt=supportVectors.begin(); svIt!=supportVectors.end(); ++svIt)
          b-=svIt->first*kernel(svIt->second, x_m);

        modelTrained=true;
      }

      /**
       * Calculate predictions based on trained model, returns the predicted
       * labels. The model has to be trained first.
       *
       * @param testData predictors, the number of columns should be the
       * same as number of features.
       */
      Labels predict(const MatrixXd& testData) const
      {
        if (!modelTrained){
          throwException("model has not been trained yet!");
        }

        if ((unsigned)testData.cols() != numberOfFeatures){
          throwException("Error happen when predicting, invalid inpute data: "
                         "expecting number of features from model: (%lu); "
                         "privided number of features from data: (%ld).\n",
                         numberOfFeatures, testData.cols());
        }

        size_t numberOfTests=testData.rows();
        Labels predictedLabels(ProblemType::Classification);
        predictedLabels.labelData.resize(numberOfTests);

        for (size_t testId=0; testId<numberOfTests; ++testId){
          double predictedLabel=b;
          for (auto svIt=supportVectors.begin(); svIt!=supportVectors.end(); ++svIt)
            predictedLabel+=svIt->first * kernel(testData.row(testId), svIt->second);
          predictedLabels.labelData(testId)=predictedLabel>0;
        }

        return predictedLabels;
      }

      /**
       * Each row is a SV.
       */
      MatrixXd getSupportVectors() const
      {
        if (!modelTrained){
          throwException("model has not been trained yet!");
        }

        size_t numberOfSVs=supportVectors.size();

        MatrixXd supportVectorData(numberOfSVs, numberOfFeatures);
        for (size_t svIndex=0; svIndex<numberOfSVs; ++svIndex)
          supportVectorData.row(svIndex)=supportVectors[svIndex].second;

        return supportVectorData;
      };

      /**
       * Clear the model.
       */
      void clear()
      {
        supportVectors.clear();
        modelTrained=false;
      }

      bool& printOptimizationProgress()
      {
        return moreOptInfo;
      }

      /**
       * Destructor.
       */
      ~SupportVectorClassifier(){}

    private:
      const Kernel kernel;
      const size_t numberOfFeatures;
      vector<std::pair<double, VectorXd> > supportVectors;
      double C;
      double tol;
      //! Indicates whether the model has been trained.
      bool modelTrained=false;
      double small=1e-5;
      double b=0.0;
      bool moreOptInfo=false;
    };
  }
}

#endif //MODEL_SUPPORT_VECTOR_CASSIFIER
