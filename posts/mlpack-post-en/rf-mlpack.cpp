#include <iostream>
// You have to include followin header files in order to run below example
#include <mlpack/core.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/split_data.hpp> // For data split
#include <mlpack/methods/random_forest/random_forest.hpp>

#define BINDING_TYPE BINDING_TYPE_CLI
#include <mlpack/core/util/mlpack_main.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::tree;


void mlpackMain()
{

    arma::mat samples;
    arma::Row<size_t> labels;

    // Load the covertype dataset
    data::Load("./dataset/covertype-small.data.csv",
               samples, true);
    data::Load("./dataset/covertype-small.labels.csv",
               labels);

    arma::mat trainData;
    arma::mat testData;
    arma::Row<size_t> trainLabel;
    arma::Row<size_t> testLabel;

    // Split the dataset into a training and test set.
    // 30% of the data for the test set.
    data::Split(samples, labels, trainData, testData, trainLabel, testLabel, 0.3);

    const size_t numClasses = arma::max(labels);
    const size_t numTree = 10;
    const size_t minLeafSize = 3;

    // An random forest mode
    RandomForest<>* rfModel = new RandomForest<>();

    Timer::Start("Training time");

    // Train an RF model
    rfModel->Train(trainData, trainLabel, numClasses, numTree, minLeafSize);

    Timer::Stop("Training time");

    Timer::Start("Testing time");

    // Predict labels of test set
    arma::Row<size_t> pred;
    rfModel->Classify(testData, pred);

    Timer::Stop("Testing time");

    const size_t correct = arma::accu(pred == testLabel);

    cout << "Accuracy on test samples: " << double(correct) / double(pred.n_elem) * 100 << endl;


    delete rfModel;

    //return 0;
}

