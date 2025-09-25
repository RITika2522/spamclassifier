package com.javabasic.spamclassifier;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class SpamClassifier {
    private Instances dataset;

    // Load dataset from CSV
    public void loadDataset(String filepath) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filepath));
            dataset = loader.getDataSet();

            // Set last attribute as class
            dataset.setClassIndex(dataset.numAttributes() - 1);
            // Print first 5 rows
            for (int i = 0; i < 5; i++) {
                System.out.println(dataset.instance(i));
            }
            System.out.println("Class attribute: " + dataset.classAttribute().name());
            System.out.println("Possible classes: " + dataset.classAttribute());

            System.out.println("Dataset loaded: " + dataset.relationName());
            System.out.println("Instances: " + dataset.numInstances());
            System.out.println("Attributes: " + dataset.numAttributes());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Getter for dataset (useful in next steps)
    public Instances getDataset() {
        return dataset;
    }
}
