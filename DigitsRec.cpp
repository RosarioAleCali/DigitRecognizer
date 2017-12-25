/* file: svm_multi_class_quality_metric_set_batch.cpp */
/*******************************************************************************
!  Copyright(C) 2014-2015 Intel Corporation. All Rights Reserved.
!
!  The source code, information  and  material ("Material") contained herein is
!  owned  by Intel Corporation or its suppliers or licensors, and title to such
!  Material remains  with Intel Corporation  or its suppliers or licensors. The
!  Material  contains proprietary information  of  Intel or  its  suppliers and
!  licensors. The  Material is protected by worldwide copyright laws and treaty
!  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!  modified, published, uploaded, posted, transmitted, distributed or disclosed
!  in any way  without Intel's  prior  express written  permission. No  license
!  under  any patent, copyright  or  other intellectual property rights  in the
!  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!  intellectual  property  rights must  be express  and  approved  by  Intel in
!  writing.
!
!  *Third Party trademarks are the property of their respective owners.
!
!  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!  this  notice or  any other notice embedded  in Materials by Intel or Intel's
!  suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!    Multi-class Support Vector Machine quality metrics example program text
!
!******************************************************************************/

/**
* <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_QUALITY_METRIC_SET_BATCH"></a>
* \example svm_multi_class_quality_metric_set_batch.cpp
*/
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <iostream>

#include "daal.h"
#include "service.h"
#include "model_file.h"

using namespace std;
using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::classifier::quality_metric;

/* Input data set parameters */
string trainDatasetFileName = "digits_tra.csv";
string trainGroundTruthFileName = "digits_tra_labels.csv";

string testDatasetFileName = "digits_tes.csv";
string testGroundTruthFileName = "digits_tes_labels.csv";

const size_t nTrainObservations = 3823;
const size_t nTestObservations = 1797;
const size_t nClasses = 10;

services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());
services::SharedPtr<svm::prediction::Batch<> > prediction1(new svm::prediction::Batch<>());

/* Model object for multi-class classifier algorithm */
services::SharedPtr<multi_class_classifier::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
services::SharedPtr<classifier::prediction::Result> predictionResult1;

/* Parameters for multi-class classifier kernel function */
kernel_function::rbf::Batch<> *rbfKernel = new kernel_function::rbf::Batch<>();
services::SharedPtr<kernel_function::KernelIface> kernel(rbfKernel);

services::SharedPtr<multi_class_classifier::quality_metric_set::ResultCollection> qualityMetricSetResult;

services::SharedPtr<NumericTable> predictedLabels;
services::SharedPtr<NumericTable> predictedLabels1;
services::SharedPtr<NumericTable> groundTruthLabels;

void trainModel();
void testModel();
void testModelQuality();
void printResults();

struct stat st;

int main(int argc, char *argv[])
{
	checkArguments(argc, argv, 4, &trainDatasetFileName, &trainGroundTruthFileName,
		&testDatasetFileName, &testGroundTruthFileName);

	rbfKernel->parameter.sigma = 1.0 / sqrt(2 * 0.001);					//22.360679774997896964091736687313;    
	training->parameter.cacheSize = 200 * 1024 * 1024;                  // default cache size in scikit
	training->parameter.kernel = kernel;
	prediction->parameter.kernel = kernel;
	prediction1->parameter.kernel = kernel;

	trainModel();

	testModel();

	testModelQuality();

	printResults();

	std::cout << std::endl << "Press any key to continue...";
	getchar();

	return 0;
}

void trainModel()
{
	/* Initialize FileDataSource<CSVFeatureManager> to retrieve input data from .csv file */
	FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
		DataSource::doAllocateNumericTable,
		DataSource::doDictionaryFromContext);
	FileDataSource<CSVFeatureManager> trainGroundTruthSource(trainGroundTruthFileName,
		DataSource::doAllocateNumericTable,
		DataSource::doDictionaryFromContext);

	/* Load data from the data files */
	trainDataSource.loadDataBlock(nTrainObservations);
	trainGroundTruthSource.loadDataBlock(nTrainObservations);

	/* Create algorithm object for multi-class SVM training */
	multi_class_classifier::training::Batch<> algorithm;

	algorithm.parameter.nClasses = nClasses;
	algorithm.parameter.training = training;
	algorithm.parameter.prediction = prediction;

	/* Pass training dataset and dependent values to the algorithm */
	algorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
	algorithm.input.set(classifier::training::labels, trainGroundTruthSource.getNumericTable());

	clock_t start, end;
	start = clock();

	/* Build multi-class SVM model */
	algorithm.compute();

	end = clock();
	printf("**********************************************************\n");
	printf("training time(ms): %.1f\n", difftime(end, start));

	/* Retrieve algorithm results */
	trainingResult = algorithm.getResult();

	/* Serialize the learned model into a disk file */
	ModelFileWriter writer("./model");
	writer.serializeToFile(trainingResult->get(classifier::training::model));
}

void testModel()
{
	/* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from .csv file */
	FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
		DataSource::doAllocateNumericTable,
		DataSource::doDictionaryFromContext);
	testDataSource.loadDataBlock(nTestObservations);

	/* Create algorithm object for prediction of multi-class SVM values */
	multi_class_classifier::prediction::Batch<> algorithm;

	algorithm.parameter.nClasses = nClasses;
	algorithm.parameter.training = training;
	algorithm.parameter.prediction = prediction;

	/* Pass testing dataset and trained model to the algorithm */
	algorithm.input.set(classifier::prediction::data, testDataSource.getNumericTable());
	algorithm.input.set(classifier::prediction::model,
		trainingResult->get(classifier::training::model));

	clock_t start, end;
	start = clock();

	/* Predict multi-class SVM values */
	algorithm.compute();
	end = clock();

	printf("prediction time(ms): %.1f\n", difftime(end, start));
	printf("**********************************************************\n");

	/* Retrieve algorithm results */
	predictionResult = algorithm.getResult();
}

void testModelQuality()
{
	/* Initialize FileDataSource<CSVFeatureManager> to retrieve ground truth data from .csv file */
	FileDataSource<CSVFeatureManager> testGroundTruth(testGroundTruthFileName,
		DataSource::doAllocateNumericTable,
		DataSource::doDictionaryFromContext);
	testGroundTruth.loadDataBlock(nTestObservations);

	/* Retrieve ground truth labels */
	groundTruthLabels = testGroundTruth.getNumericTable();

	/* Retrieve predicted labels */
	predictedLabels = predictionResult->get(classifier::prediction::prediction);

	/* Create quality metric set object to compute quality metrics of the multi-class classifier algorithm */
	multi_class_classifier::quality_metric_set::Batch qualityMetricSet(nClasses);

	services::SharedPtr<multiclass_confusion_matrix::Input> input =
		qualityMetricSet.getInputDataCollection()->getInput(multi_class_classifier::quality_metric_set::confusionMatrix);

	input->set(multiclass_confusion_matrix::predictedLabels, predictedLabels);
	input->set(multiclass_confusion_matrix::groundTruthLabels, groundTruthLabels);

	/* Compute quality metrics */
	qualityMetricSet.compute();

	/* Retrieve quality metrics */
	qualityMetricSetResult = qualityMetricSet.getResultCollection();
}

void printResults()
{
	/* Print classification results */
	//printNumericTables<int, double>(groundTruthLabels.get(), predictedLabels.get(),
	//	"Ground truth", "Classification results",
	//	"SVM classification results (first 20 observations):", 20);
	/* Print quality metrics */
	services::SharedPtr<multiclass_confusion_matrix::Result> qualityMetricResult =
		qualityMetricSetResult->getResult(multi_class_classifier::quality_metric_set::confusionMatrix);
	printNumericTable(qualityMetricResult->get(multiclass_confusion_matrix::confusionMatrix), "Confusion matrix:");

	BlockDescriptor<> block;
	services::SharedPtr<NumericTable> qualityMetricsTable = qualityMetricResult->get(multiclass_confusion_matrix::multiClassMetrics);
	qualityMetricsTable->getBlockOfRows(0, 1, readOnly, block);
	// double *qualityMetricsData = block.getBlockPtr();
	float *qualityMetricsData = block.getBlockPtr();
	std::cout << "Average accuracy: " << qualityMetricsData[multiclass_confusion_matrix::averageAccuracy] << std::endl;
	std::cout << "Error rate:       " << qualityMetricsData[multiclass_confusion_matrix::errorRate] << std::endl;
	std::cout << "Micro precision:  " << qualityMetricsData[multiclass_confusion_matrix::microPrecision] << std::endl;
	std::cout << "Micro recall:     " << qualityMetricsData[multiclass_confusion_matrix::microRecall] << std::endl;
	std::cout << "Micro F-score:    " << qualityMetricsData[multiclass_confusion_matrix::microFscore] << std::endl;
	std::cout << "Macro precision:  " << qualityMetricsData[multiclass_confusion_matrix::macroPrecision] << std::endl;
	std::cout << "Macro recall:     " << qualityMetricsData[multiclass_confusion_matrix::macroRecall] << std::endl;
	std::cout << "Macro F-score:    " << qualityMetricsData[multiclass_confusion_matrix::macroFscore] << std::endl;
	qualityMetricsTable->releaseBlockOfRows(block);
}


