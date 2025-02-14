package com.sergosoft.neural.perceptron;

import java.util.Random;

public class Perceptron {

    static final int INPUT_NEURONS = 4;
    static final int HIDDEN_NEURONS = 4;
    static final int TRAINING_EPOCHS = 10000;
    static final double LEARNING_RATE = 0.1;

    static double[][] weightsInputToHidden = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS]; // Includes bias
    static double[] weightsHiddenToOutput = new double[HIDDEN_NEURONS + 1]; // Includes bias

    public static void main(String[] args) {
        initializeWeights();

        int[][] trainingInputs = {
                {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
                {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
                {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
        };

        int[] expectedOutputs = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};

        train(trainingInputs, expectedOutputs);

        System.out.println("Trained weights:");
        for (double weight : weightsHiddenToOutput) {
            System.out.printf("%.5f ", weight);
        }
        System.out.println();

        System.out.println("Testing:");
        for (int[] input : trainingInputs) {
            System.out.printf("Input: %s -> Output: %.5f\n", java.util.Arrays.toString(input), predict(input));
        }
    }

    static void initializeWeights() {
        Random random = new Random();
        for (int inputIdx = 0; inputIdx <= INPUT_NEURONS; inputIdx++) {
            for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                weightsInputToHidden[inputIdx][hiddenIdx] = random.nextDouble() * 2 - 1;
            }
        }
        for (int hiddenIdx = 0; hiddenIdx <= HIDDEN_NEURONS; hiddenIdx++) {
            weightsHiddenToOutput[hiddenIdx] = random.nextDouble() * 2 - 1;
        }
    }

    static void train(int[][] trainingInputs, int[] expectedOutputs) {
        for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
            for (int sampleIdx = 0; sampleIdx < trainingInputs.length; sampleIdx++) {
                double[] hiddenLayerOutputs = new double[HIDDEN_NEURONS];

                // Forward pass: Input to Hidden Layer
                for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                    double activationSum = weightsInputToHidden[INPUT_NEURONS][hiddenIdx]; // Bias
                    for (int inputIdx = 0; inputIdx < INPUT_NEURONS; inputIdx++) {
                        activationSum += trainingInputs[sampleIdx][inputIdx] * weightsInputToHidden[inputIdx][hiddenIdx];
                    }
                    hiddenLayerOutputs[hiddenIdx] = sigmoid(activationSum);
                }

                // Forward pass: Hidden to Output Layer
                double outputLayerSum = weightsHiddenToOutput[HIDDEN_NEURONS]; // Bias
                for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                    outputLayerSum += hiddenLayerOutputs[hiddenIdx] * weightsHiddenToOutput[hiddenIdx];
                }
                double predictedOutput = sigmoid(outputLayerSum);

                // Backpropagation
                double outputError = expectedOutputs[sampleIdx] - predictedOutput;
                double outputGradient = outputError * sigmoidDerivative(predictedOutput);

                // Calculate hidden layer error gradients
                double[] hiddenGradients = new double[HIDDEN_NEURONS];
                for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                    hiddenGradients[hiddenIdx] = outputGradient * weightsHiddenToOutput[hiddenIdx] * sigmoidDerivative(hiddenLayerOutputs[hiddenIdx]);
                }

                // Update weights: Hidden to Output
                for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                    weightsHiddenToOutput[hiddenIdx] += LEARNING_RATE * outputGradient * hiddenLayerOutputs[hiddenIdx];
                }
                weightsHiddenToOutput[HIDDEN_NEURONS] += LEARNING_RATE * outputGradient; // Bias update

                // Update weights: Input to Hidden
                for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
                    for (int inputIdx = 0; inputIdx < INPUT_NEURONS; inputIdx++) {
                        weightsInputToHidden[inputIdx][hiddenIdx] += LEARNING_RATE * hiddenGradients[hiddenIdx] * trainingInputs[sampleIdx][inputIdx];
                    }
                    weightsInputToHidden[INPUT_NEURONS][hiddenIdx] += LEARNING_RATE * hiddenGradients[hiddenIdx]; // Bias update
                }
            }
        }
    }

    static double predict(int[] input) {
        double[] hiddenLayerOutputs = new double[HIDDEN_NEURONS];

        // Forward pass: Input to Hidden
        for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
            double activationSum = weightsInputToHidden[INPUT_NEURONS][hiddenIdx]; // Bias
            for (int inputIdx = 0; inputIdx < INPUT_NEURONS; inputIdx++) {
                activationSum += input[inputIdx] * weightsInputToHidden[inputIdx][hiddenIdx];
            }
            hiddenLayerOutputs[hiddenIdx] = sigmoid(activationSum);
        }

        // Forward pass: Hidden to Output
        double outputLayerSum = weightsHiddenToOutput[HIDDEN_NEURONS]; // Bias
        for (int hiddenIdx = 0; hiddenIdx < HIDDEN_NEURONS; hiddenIdx++) {
            outputLayerSum += hiddenLayerOutputs[hiddenIdx] * weightsHiddenToOutput[hiddenIdx];
        }

        return sigmoid(outputLayerSum);
    }

    static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
}
