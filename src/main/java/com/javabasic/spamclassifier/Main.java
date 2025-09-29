package com.javabasic.spamclassifier;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) {
        // Create classifier object
        SpamClassifier classifier = new SpamClassifier();

        // Load dataset
        classifier.loadDataset("data/spambase.csv");

        // Apply Normalization
        Instances normalizedData = classifier.normalizeData(classifier.getDataset());
        System.out.println("Instances after normalization: " + normalizedData.numInstances());

        // Apply Standardization
        Instances standardizedData = classifier.standardizeData(classifier.getDataset());
        System.out.println("Instances after standardization: " + standardizedData.numInstances());
    }
}
