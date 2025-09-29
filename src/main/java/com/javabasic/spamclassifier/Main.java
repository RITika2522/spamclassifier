package com.javabasic.spamclassifier;

public class Main {
    public static void main(String[] args) {
        try {
            // Create classifier object
            SpamClassifier classifier = new SpamClassifier();

            // Load ARFF dataset
            classifier.loadDataset("data/spambase.arff");

            // Train and evaluate using 10-fold CV
            classifier.trainNaiveBayesWithCV();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
