package com.sergosoft.neural.perceptron;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class PerceptronTest {

    private Perceptron perceptron;

    @BeforeEach
    void setUp() {
        perceptron = new Perceptron();
        Perceptron.initializeWeights(); // Ensure weights are reset before each test
    }

    @Test
    void testXorFunctionalityAfterTraining() {
        int[][] trainingInputs = {
                {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
                {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
                {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
        };

        int[] expectedOutputs = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};

        Perceptron.train(trainingInputs, expectedOutputs);

        for (int i = 0; i < trainingInputs.length; i++) {
            double prediction = Perceptron.predict(trainingInputs[i]);
            int binaryOutput = prediction >= 0.5 ? 1 : 0; // Convert sigmoid output to binary
            assertEquals(expectedOutputs[i], binaryOutput,
                    "Failed for input: " + java.util.Arrays.toString(trainingInputs[i]));
        }
    }

    @Test
    void testSigmoidFunction() {
        assertEquals(0.5, Perceptron.sigmoid(0), 0.0001);
        assertTrue(Perceptron.sigmoid(10) > 0.9);
        assertTrue(Perceptron.sigmoid(-10) < 0.1);
    }

    @Test
    void testSigmoidDerivative() {
        double sigmoidValue = Perceptron.sigmoid(0.5);
        assertEquals(sigmoidValue * (1 - sigmoidValue), Perceptron.sigmoidDerivative(sigmoidValue), 0.0001);
    }
}
