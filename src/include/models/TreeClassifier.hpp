#ifndef MODEL_TREE_CLASSIFIER
#define MODEL_TREE_CLASSIFIER

#include "../core/Definitions.hpp"
#include "../core/Utilities.hpp"
#include "../internal/_Builder.hpp"
#include "../internal/_Tree.hpp"

namespace CPPLearn
{
  namespace Models
  {
    template<double (*ImpurityRule)(const vector<long>&)>
    class TreeClassifier {
    public:
      static const ProblemType ModelType = ProblemType::Classification;
      static constexpr const char* ModelName="TreeClassifier";
      static constexpr double (*LossFunction)(const Labels&, const Labels&)=
        Utilities::classificationZeroOneLossFunction;

      using Criterion = _Criterion<ImpurityRule>;

      TreeClassifier(const long numberOfFeatures, const long numberOfClasses,
                     const long minSamplesInALeaf, const long minSamplesInANode,
                     const long maxDepth, const long maxNumberOfLeafNodes=-1) :
        _numberOfFeatures{numberOfFeatures}, _numberOfClasses{numberOfClasses},
        _minSamplesInALeaf{minSamplesInALeaf}, _minSamplesInANode{minSamplesInANode},
        _maxDepth{maxDepth}, _maxNumberOfLeafNodes{maxNumberOfLeafNodes} { }

      void train(const MatrixXd& trainData, const Labels& trainLabels)
      {
        if (trainLabels._labelType != ProblemType::Classification){
          throwException("Error happen when training %s model: "
                         "Input labelType must be Classification!\n", ModelName);
        }

        const VectorXd& labelData=trainLabels._labelData;

        if (trainData.cols() != _numberOfFeatures){
          throwException("Error happen when training %s model, "
                         "invalid inpute data: "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         ModelName, _numberOfFeatures, trainData.cols());
        }

        if (trainData.rows() != labelData.size()){
          throwException("data and label size mismatch! "
                         "number of data: (%ld), "
                         "number of labels: (%ld), ",
                         trainData.rows(), labelData.size());
        }

        vector<long> labelsCount(_numberOfClasses, 0);
        const long numberOfTrainData = trainData.rows();

        for (long labelIndex=0; labelIndex<numberOfTrainData; ++labelIndex){
          if (labelData(labelIndex) < 0 || labelData(labelIndex) >= _numberOfClasses){
            throwException("Error happen when training multiclass classifier: "
                           "number of classes is (%ld), "
                           "thus  I am expecting label range between (0) and (%ld)! "
                           "the (%ld)th label is (%f)!",
                           _numberOfClasses, _numberOfClasses-1,
                           labelIndex, labelData(labelIndex));
          }
          ++labelsCount[static_cast<long>(labelData(labelIndex))];
        }

        for (long classIndex=0; classIndex<_numberOfClasses; ++classIndex)
          if (labelsCount[classIndex] == 0){
            throwException("Error happened from Multiclass classifer:\n"
                           "None of the provided training data is classified to class (%ld). "
                           "You need to consider reducing the input number of classes!\n",
                           classIndex);
          }

        if(_verbose == VerboseFlag::Verbose){
          cout<<"In Mulicalss Classifier, Multiclass training input label summary: "<<endl;
          cout<<"number of classes: "<<_numberOfClasses<<endl;
          cout<<"number of training data: "<<numberOfTrainData<<endl;
          for (long classIndex=0l; classIndex<_numberOfClasses; ++classIndex)
            cout<<"number of training data labeled "<<classIndex<<": "<<
              labelsCount[classIndex]<<endl;
        }

        _tree._numberOfClasses = _numberOfClasses;
        _tree._numberOfFeatures = _numberOfFeatures;

        Criterion criterion{_numberOfClasses};
        //Splitter splitter{criterion};

        std::unique_ptr<_BuilderBase> builder=nullptr;
        if (_maxNumberOfLeafNodes < 0)
          builder=std::make_unique<_DepthFirstBuilder<Criterion> >(_minSamplesInALeaf,
                                                                  _minSamplesInANode,
                                                                  _maxDepth,
                                                                  &criterion);
        else
          builder=std::make_unique<_BestFirstBuilder<Criterion> >(_minSamplesInALeaf,
                                                                 _minSamplesInANode,
                                                                 _maxDepth,
                                                                 _maxNumberOfLeafNodes,
                                                                 &criterion);

        try {
          builder->build(trainData, labelData, &_tree);
        }
        catch (...) {
          cout<<"exception caught when training tree classifier: "<<endl;
          throw;
        }

        _modelTrained=true;
      }

      Labels predict(const MatrixXd& testData) const
      {
        if (!_modelTrained){
          throwException("Error happen when predicting with %s model: "
                         "Model has not been trained yet!", ModelName);
        }

        if (testData.cols() != _numberOfFeatures){
          throwException("Error happen when predicting with %s model: "
                         "Invalid inpute data, "
                         "expecting number of features from model: (%ld); "
                         "privided number of features from data: (%ld).\n",
                         ModelName, _numberOfFeatures, testData.cols());
        }

        long numberOfTests=testData.rows();
        Labels predictedLabels{ProblemType::Classification};
        predictedLabels._labelData.resize(numberOfTests);

        for (long rowIndex=0; rowIndex<numberOfTests; ++rowIndex){
          Map<const VectorXd> instance(&testData(rowIndex, 0), _numberOfFeatures);
          predictedLabels._labelData(rowIndex) = predictOne(instance);
        }

        return predictedLabels;
      }

      double predictOne(const VectorXd& instance) const
      {
        vector<long> labelsCount= _tree.predictOne(instance);
        assert(static_cast<long>(labelsCount.size())==_numberOfClasses);
        auto it=std::max_element(std::begin(labelsCount), std::end(labelsCount));
        //need to take care of bias term
        long label =(it-std::begin(labelsCount));

        return static_cast<double>(label);
      }

      const long& getNumberOfFeatures() const
      {
        return _numberOfFeatures;
      }

      /**
       * Clear the model.
       */
      void clear()
      {
        _tree.reset();
        _modelTrained=false;
      }

      VerboseFlag& whetherVerbose()
      {
        return _verbose;
      }

    private:
      const long _numberOfFeatures;
      const long _numberOfClasses;
      const long _minSamplesInALeaf;
      const long _minSamplesInANode;
      const long _maxDepth;
      const long _maxNumberOfLeafNodes;
      bool _modelTrained=false;
      VerboseFlag _verbose = VerboseFlag::Quiet;
      _Tree _tree;
    };
  }
}

#endif //MODEL_TREE_CLASSIFIER
