package com.javabasic.spamclassifier;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.util.Random;

public class SpamClassifier {
    private Instances dataset;
    private Classifier model;

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

    // Load trained model from file
    public void loadModel(String modelPath) {
        try {
            model = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("Model loaded from " + modelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Load ARFF header (to create new instances later)
    public void setDatasetHeader(Instances data) {
        this.dataset = data;
        this.dataset.setClassIndex(dataset.numAttributes() - 1);
    }

    // Classify a single instance
    public String classifyEmail(double[] values) {
        try {
            // Create new instance with same number of attributes
            Instance newInst = new weka.core.DenseInstance(dataset.numAttributes());
            newInst.setDataset(dataset);

            // Fill feature values (excluding class)
            for (int i = 0; i < dataset.numAttributes() - 1; i++) {
                newInst.setValue(i, values[i]);
            }

            // Predict
            double prediction = model.classifyInstance(newInst);
            String className = dataset.classAttribute().value((int) prediction);

            return className;
        } catch (Exception e) {
            e.printStackTrace();
            return "Error";
        }
    }

    // Classify email from a CSV row (string of comma-separated values)
    public String classifyEmailFromCSV(String csvRow) {
        try {
            String[] parts = csvRow.split(",");

            // Expect dataset.numAttributes() - 1 values (exclude class label)
            int expected = dataset.numAttributes() - 1;
            if (parts.length != expected) {
                return "Error: Expected " + expected + " values, but got " + parts.length;
            }

            double[] values = new double[expected];
            for (int i = 0; i < expected; i++) {
                values[i] = Double.parseDouble(parts[i]);
            }

            // Create instance
            DenseInstance newInst = new DenseInstance(dataset.numAttributes());
            newInst.setDataset(dataset);

            for (int i = 0; i < expected; i++) {
                newInst.setValue(i, values[i]);
            }

            // Predict
            double prediction = model.classifyInstance(newInst);
            return dataset.classAttribute().value((int) prediction);

        } catch (Exception e) {
            e.printStackTrace();
            return "Error";
        }
    }

    public Instances getDataset() {
        return dataset;
    }
}
