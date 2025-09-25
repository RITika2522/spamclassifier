package com.javabasic.spamclassifier;

public class Main {
    public static void main(String[] args) {
        // Create classifier object
        SpamClassifier classifier = new SpamClassifier();

        // Load dataset
        classifier.loadDataset("data/spambase.csv");

        // Later steps: classifier.trainModel(), classifier.evaluateModel(), etc.
    }
}
