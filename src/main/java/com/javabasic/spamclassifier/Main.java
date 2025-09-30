package com.javabasic.spamclassifier;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        try {
            SpamClassifier classifier = new SpamClassifier();

            // 1. Load dataset
            classifier.loadDataset("data/spambase.arff");

            // 2. Train + evaluate Naive Bayes
            NaiveBayes nb = new NaiveBayes();
            classifier.trainAndEvaluate(nb, "Naive Bayes");

            // 3. Save trained model
            classifier.saveModel(nb, "spam.model");

            // 4. Load model back
            classifier.loadModel("spam.model");
            classifier.setDatasetHeader(classifier.getDataset()); // needed for new instances

            // 5. CLI input
            Scanner scanner = new Scanner(System.in);
            System.out.println("\n=== Spam Email Classifier CLI ===");
            System.out.println("Paste a CSV row (without class label), or type 'exit' to quit.");

            while (true) {
                System.out.print("\nEnter CSV row: ");
                String input = scanner.nextLine();

                if (input.equalsIgnoreCase("exit")) {
                    System.out.println("Exiting...");
                    break;
                }

                // Classify input row
                String result = classifier.classifyEmailFromCSV(input);
                System.out.println("Prediction: " + result);
            }

            scanner.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
