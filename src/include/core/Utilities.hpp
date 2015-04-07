// -*- C++ -*-
#ifndef UTILITIES_H
#define UTILITIES_H

#include "Definitions.hpp"
#include <Eigen/Eigenvalues>
#include <sys/stat.h>

namespace Utilities {

  enum class TrimStyle {DontTrim, Trim};

  // retrieved from http://oxaric.wordpress.com/2008/11/23/3-simple-c-functions/
  std::string trim( std::string line ) {
    if ( line.empty() ) {
      return "";
    }
    int string_size = (int)(line.length());
    int beginning_of_string = 0;
    // the minus 1 is needed to start at the first character
    // and skip the string delimiter
    int end_of_string = string_size - 1;
    bool encountered_characters = false;
    // find the start of chracters in the string
    while ( (beginning_of_string < string_size) && (!encountered_characters) ) {
      // if a space or tab was found then ignore it
      if ( (line[ beginning_of_string ] != ' ') &&
           (line[ beginning_of_string ] != '\t') ) {
        encountered_characters = true;
      } else {
        beginning_of_string++;
      }
    }
    // test if no characters were found in the string
    if ( beginning_of_string == string_size ) {
      return "";
    }
    encountered_characters = false;
    // find the character in the string
    while ( (end_of_string > beginning_of_string) &&
            (!encountered_characters) ) {
      // if a space or tab was found then ignore it
      if ( (line[ end_of_string ] != ' ') && (line[ end_of_string ] != '\t') ) {
        encountered_characters = true;
      } else {
        end_of_string--;
      }
    }
    // return the original string with all whitespace removed from
    // its beginning and end
    // + 1 at the end to add the space for the string delimiter
    return line.substr( beginning_of_string,
                        end_of_string - beginning_of_string + 1 );
  }

  std::vector<std::string> tokenize(const std::string& str, const std::string& delimiters,
                                    const TrimStyle doTrim) {
    // Skip delimiters at beginning.
    std::vector<std::string> tokens;
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      // Found a token, add it to the vector.
      std::string candidate = str.substr(lastPos, pos - lastPos);
      if (doTrim == TrimStyle::Trim) {
        candidate = trim(candidate);
      }
      tokens.push_back(candidate);
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
    }
    return tokens;
  }

  template <class T>
  T
  convertString(const string & str) {
    std::stringstream ss;
    ss << str;
    T returnVal;
    ss >> returnVal;
    return returnVal;
  }

  template <>
  float
  convertString<float>(const string & str) {
    if (strcmp(str.c_str(), "nan") == 0) {
      //printf("Parsed a nan\n");
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      std::stringstream ss;
      ss << str;
      float returnVal;
      ss >> returnVal;
      return returnVal;
    }
  }

  template <>
  double
  convertString<double>(const string & str) {
    if (strcmp(str.c_str(), "nan") == 0) {
      //printf("Parsed a nan\n");
      return std::numeric_limits<double>::quiet_NaN();
    } else {
      std::stringstream ss;
      ss << str;
      double returnVal;
      ss >> returnVal;
      return returnVal;
    }
  }

  void
  directoryCreator(const string & fullPath,
                   const bool createNewDirectory = true,
                   const VerboseFlag verboseFlag = VerboseFlag::Verbose){

    // split the fullPath by slashes
    vector<string> tokens = Utilities::tokenize(fullPath, "/",
                                                Utilities::TrimStyle::Trim);

    string incrementalPath;
    for (unsigned int tokenIndex = 0; tokenIndex < tokens.size(); ++tokenIndex) {

      // keep building the incrementalPath
      if (tokenIndex > 0) {
        incrementalPath += "/";
      }
      incrementalPath += tokens[tokenIndex];

      std::ifstream ifile(incrementalPath.c_str());
      if ((bool)ifile == false && createNewDirectory == true) {
        if (verboseFlag == VerboseFlag::Verbose){
          printf("Did not find a directory at %s, creating one.\n",
                 incrementalPath.c_str());
        }
        int status =
          mkdir(incrementalPath.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status != 0 && verboseFlag == VerboseFlag::Verbose) {
          fprintf(stderr, "Problem creating directory %s\n",
                  incrementalPath.c_str());
        }
      } else if ((bool)ifile == false && createNewDirectory == false
                 && verboseFlag == VerboseFlag::Verbose){
        printf("Couldn't find the directory at %s that you specified,"
               "and you said not to make it\n",incrementalPath.c_str());
      } else if (verboseFlag == VerboseFlag::Verbose){
        printf("Found a directory already existing at %s, not creating one.\n",
               incrementalPath.c_str());
      }
    }
  }


  void createCPPLearnDataFileFromLibsvmFormat(const string& libsvmFormatFileName,
                                             const string& cpplearnFileName,
                                             const VerboseFlag verboseFlag=VerboseFlag::Verbose){
    ifstream inputFile;
    inputFile.open(libsvmFormatFileName);

    if (!inputFile.is_open()){
      throwException("Unable to open file at %s\n", libsvmFormatFileName.c_str());
    }

    vector<int> labels;
    vector<vector<double> > data;
    string line;
    size_t numberOfFeatures=0;

    if (verboseFlag==VerboseFlag::Verbose)
      cout<<"Begin read data from: "<<libsvmFormatFileName<<endl;

    while (inputFile.good()) {
      getline (inputFile,line);
      if (line == "") break;
      vector<double> instance;
      vector<string> tokens = Utilities::tokenize(line, " ", Utilities::TrimStyle::Trim);
      labels.push_back(atof(tokens[0].c_str()));

      size_t featIndex = 0;
      for (size_t tokIndex=1; tokIndex<tokens.size(); ++tokIndex){
        vector<string> subtokens = Utilities::tokenize(tokens[tokIndex], ":", TrimStyle::Trim);
        size_t index=atoi(subtokens[0].c_str());
        double value=atof(subtokens[1].c_str());
        while (featIndex<index-1){
          instance.push_back(0);
          ++featIndex;
        }

        instance.push_back(value);
        ++featIndex;
      }

      if (featIndex>numberOfFeatures) numberOfFeatures=featIndex;
      data.push_back(std::move(instance));
    }

    size_t numberOfData=labels.size();
    FILE* outfile=fopen(cpplearnFileName.c_str(),"w");
    if (!outfile){
      throwException("Unable to open output file at %s\n",
                     cpplearnFileName.c_str());
    }

    if (verboseFlag==VerboseFlag::Verbose)
      cout<<"Finsh reading data, begin write data to file: "<<cpplearnFileName<<endl;

    fprintf(outfile,"%lu %lu\n", numberOfData, numberOfFeatures);

    for (size_t dataIndex=0; dataIndex<numberOfData; ++dataIndex){
      fprintf(outfile,"%+d ", labels[dataIndex]);
      for (size_t featIndex=0; featIndex<numberOfFeatures; ++featIndex){
        if(featIndex<data[dataIndex].size())
          fprintf(outfile,"%20.10e", data[dataIndex][featIndex]);
        else
          fprintf(outfile,"%20.10e ", 0.0);
      }
      fprintf(outfile, "\n");
    }

    fclose(outfile);

    if (verboseFlag==VerboseFlag::Verbose){
      cout<<"Finish converting libsvm format data to cpplearn format data"<<endl;
      cout<<"Data is written to "<<cpplearnFileName<<endl;
      cout<<"Summary:"<<endl;
      cout<<"Number of data = "<<numberOfData<<endl;
      cout<<"Number of features = "<<numberOfFeatures<<endl;
    }
  }

  std::pair<MatrixXd, VectorXd>
  readCPPLearnDataFile(const string& filename){
    ifstream file;
    file.open(filename);
    if (!file.is_open()){
      throwException("Unable to open output file at %s\n", filename.c_str());
    }

    string line;
    getline(file, line);
    vector<string> tokens = tokenize(line, " ", TrimStyle::Trim);

    if (tokens.size() != 2){
      throwException("File %s is in the wrong format. "
                     "First line needs to specify the number of data "
                     "and number of features, seperated by space!",
                     filename.c_str());
    }

    size_t numberOfData = atoi(tokens[0].c_str());
    size_t numberOfFeatures = atoi(tokens[1].c_str());

    MatrixXd data(numberOfData, numberOfFeatures);
    VectorXd labels(numberOfData);

    for (size_t lineIndex=1; lineIndex<=numberOfData; ++lineIndex){
      getline(file, line);
      if (line == ""){
        throwException("line number (%lu) is an empty line in the middel, "
                       "which is forbidden by CPPLearn data format\n",
                       lineIndex);
      }

      vector<string> tokens = tokenize(line, " ", TrimStyle::Trim);
      if (tokens.size() != numberOfFeatures+1){
        throwException("In line %lu, expected number of column is (%lu), "
                       "while the provide number of colum is (%lu)\n",
                       lineIndex, numberOfFeatures+1, tokens.size());
      }

      labels(lineIndex-1)=atof(tokens[0].c_str());
      for (size_t tokIndex=1; tokIndex<=numberOfFeatures; ++tokIndex)
        data(lineIndex-1, tokIndex-1) = atof(tokens[tokIndex].c_str());
    }
    getline(file, line);
    if (line != "")
      printf("Number of data provided seems more than the number of data specified,"
             "will only use the first (%lu) data.\n", numberOfData);

    return std::make_pair(data, labels);
  }
}

#endif  // UTILITIES_H
