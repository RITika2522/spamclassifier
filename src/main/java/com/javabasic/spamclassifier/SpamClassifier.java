package com.javabasic.spamclassifier;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

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

    // Getter for dataset
    public Instances getDataset() {
        return dataset;
    }

    // 🔹 Apply Normalization
    public Instances normalizeData(Instances data) {
        try {
            Normalize normalize = new Normalize();
            normalize.setInputFormat(data);
            Instances newData = Filter.useFilter(data, normalize);
            System.out.println("Normalization complete.");
            return newData;
        } catch (Exception e) {
            e.printStackTrace();
            return data;
        }
    }

    // 🔹 Apply Standardization
    public Instances standardizeData(Instances data) {
        try {
            Standardize standardize = new Standardize();
            standardize.setInputFormat(data);
            Instances newData = Filter.useFilter(data, standardize);
            System.out.println("Standardization complete.");
            return newData;
        } catch (Exception e) {
            e.printStackTrace();
            return data;
        }
    }
}
