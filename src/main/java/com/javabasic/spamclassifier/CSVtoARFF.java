package com.javabasic.spamclassifier;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

public class CSVtoARFF {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/spambase.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/spambase.arff"));
        saver.writeBatch();

        System.out.println("CSV converted to ARFF successfully!");
    }
}
