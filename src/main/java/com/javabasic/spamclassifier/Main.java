package com.javabasic.spamclassifier;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.Classifier;

public class Main {
    public static void main(String[] args) {
        try {
            // Create classifier object
            SpamClassifier classifier = new SpamClassifier();

            // Load ARFF dataset
            classifier.loadDataset("data/spambase.arff");

            // Run Naive Bayes
            classifier.trainAndEvaluate(new NaiveBayes(), "Naive Bayes");

            // Run Random Forest
            classifier.trainAndEvaluate(new RandomForest(), "Random Forest");

            // Run SVM (SMO in Weka)
            classifier.trainAndEvaluate(new SMO(), "SVM (SMO)");

            // === Save the best model (example: RandomForest) ===
            String modelPath = "models/spam_randomforest.model";
            classifier.saveModel(new RandomForest(), modelPath);

            // === Load model back ===
            Classifier loadedModel = classifier.loadModel(modelPath);

            if (loadedModel != null) {
                System.out.println("Successfully reloaded trained model: " + loadedModel.getClass().getSimpleName());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
