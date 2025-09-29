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

    // Train Naive Bayes with 10-fold cross-validation
    public void trainNaiveBayesWithCV() {
        try {
            classifier = new NaiveBayes();
            classifier.buildClassifier(dataset);

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(classifier, dataset, 10, new Random(1));

            System.out.println("\n=== 10-Fold Cross-Validation Results ===");
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("Precision  = " + eval.precision(1));
            System.out.println("Recall     = " + eval.recall(1));
            System.out.println("F1 Score   = " + eval.fMeasure(1));
            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Classifier getClassifier() {
        return classifier;
    }
}
