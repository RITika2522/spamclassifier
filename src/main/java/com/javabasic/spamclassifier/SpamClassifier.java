package com.javabasic.spamclassifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class SpamClassifier {
    private Instances dataset;
    private Classifier classifier;

    // Load dataset from ARFF
    public void loadDataset(String filePath) {
        try {
            DataSource source = new DataSource(filePath);
            dataset = source.getDataSet();

            // Set class index to last attribute
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
            System.out.println("Dataset loaded: " + filePath);
            System.out.println("Instances: " + dataset.numInstances());
            System.out.println("Attributes: " + dataset.numAttributes());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Train and evaluate with 10-fold cross-validation
    public void trainAndEvaluate(Classifier cls, String classifierName) {
        try {
            Evaluation eval = new Evaluation(dataset);

            // Perform 10-fold cross-validation
            eval.crossValidateModel(cls, dataset, 10, new Random(1));

            System.out.println("\n=== Results for " + classifierName + " ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Instances getDataset() {
        return dataset;
    }
}
