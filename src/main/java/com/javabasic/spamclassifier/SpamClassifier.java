package com.javabasic.spamclassifier;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.util.Random;

public class SpamClassifier {
    private Instances dataset;

    // Load dataset from ARFF
    public void loadDataset(String filepath) {
        try {
            DataSource source = new DataSource(filepath);
            dataset = source.getDataSet();

            // Set class index to the last attribute
            dataset.setClassIndex(dataset.numAttributes() - 1);

            System.out.println("Dataset loaded: " + dataset.relationName());
            System.out.println("Instances: " + dataset.numInstances());
            System.out.println("Attributes: " + dataset.numAttributes());
            System.out.println("Class attribute: " + dataset.classAttribute().name());
            System.out.println("Possible classes: " + dataset.classAttribute());
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

    // Train final model on full dataset and save it
    public void saveModel(Classifier cls, String modelPath) {
        try {
            cls.buildClassifier(dataset);  // Train on full dataset
            SerializationHelper.write(modelPath, cls);
            System.out.println("Model saved to: " + modelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Load model from file
    public Classifier loadModel(String modelPath) {
        try {
            Classifier cls = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Model loaded from: " + modelPath);
            return cls;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public Instances getDataset() {
        return dataset;
    }
}
