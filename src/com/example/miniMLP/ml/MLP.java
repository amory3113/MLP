package com.example.miniMLP.ml;

import java.io.Serializable;
import java.util.Random;

public class MLP implements Serializable {
    private static final long serialVersionUID = 1L;
    private int inputSize, hiddenSize, outputSize;
    private float[][] w1;
    private float[] b1;
    private float[][] w2;
    private float[] b2;

    private float regLambda = 0.001f;
    private float dropoutRate = 0.2f;

    private transient Random rnd = new Random(42);

    private float entropyThreshold = 0.7f;

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        w1 = new float[inputSize][hiddenSize];
        b1 = new float[hiddenSize];
        w2 = new float[hiddenSize][outputSize];
        b2 = new float[outputSize];

        initWeights(w1);
        initWeights(w2);
    }

    private void initWeights(float[][] matrix) {
        float scale = (float) Math.sqrt(2.0 / (matrix.length + matrix[0].length));
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = (rnd.nextFloat() * 2 - 1) * scale;
            }
        }
    }

    public void train(float[][] inputs, float[][] targets, int epochs, float lr) {
        int n = inputs.length;

        // Добавляем early stopping
        float bestLoss = Float.MAX_VALUE;
        float[][] bestW1 = new float[inputSize][hiddenSize];
        float[] bestB1 = new float[hiddenSize];
        float[][] bestW2 = new float[hiddenSize][outputSize];
        float[] bestB2 = new float[outputSize];

        int patience = 30; // Количество эпох без улучшения
        int noImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            float sumLoss = 0f;

            // Shuffle data
            shuffleData(inputs, targets);

            for (int i = 0; i < n; i++) {
                sumLoss += trainOnExample(inputs[i], targets[i], lr);
            }

            float avgLoss = sumLoss / n;
            System.out.println("Epoch " + epoch + " - Loss: " + avgLoss);

            // Check for improvement
            if (avgLoss < bestLoss) {
                bestLoss = avgLoss;
                noImprovement = 0;

                // Save best weights
                copyWeights(w1, bestW1);
                copyWeights(w2, bestW2);
                System.arraycopy(b1, 0, bestB1, 0, b1.length);
                System.arraycopy(b2, 0, bestB2, 0, b2.length);
            } else {
                noImprovement++;
                if (noImprovement >= patience) {
                    System.out.println("Early stopping at epoch " + epoch);
                    break;
                }
            }

            // Learning rate decay
            if (epoch % 100 == 0 && epoch > 0) {
                lr *= 0.9f;
            }
        }

        // Restore best weights
        w1 = bestW1;
        b1 = bestB1;
        w2 = bestW2;
        b2 = bestB2;
    }

    private void shuffleData(float[][] inputs, float[][] targets) {
        for (int i = 0; i < inputs.length; i++) {
            int j = rnd.nextInt(inputs.length);
            // Swap inputs
            float[] tempInput = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = tempInput;

            // Swap targets
            float[] tempTarget = targets[i];
            targets[i] = targets[j];
            targets[j] = tempTarget;
        }
    }

    private void copyWeights(float[][] source, float[][] dest) {
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, dest[i], 0, source[i].length);
        }
    }

    private float trainOnExample(float[] input, float[] target, float lr) {
        // Forward pass with dropout
        float[] hiddenRaw = new float[hiddenSize];
        float[] hidden = new float[hiddenSize];
        boolean[] dropoutMask = new boolean[hiddenSize];

        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hiddenRaw[j] = sum;
            hidden[j] = relu(sum);

            // Apply dropout during training
            dropoutMask[j] = rnd.nextFloat() > dropoutRate;
            if (!dropoutMask[j]) {
                hidden[j] = 0;
            } else {
                // Scale to maintain same expected value
                hidden[j] /= (1.0f - dropoutRate);
            }
        }

        float[] outputRaw = new float[outputSize];
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < outputSize; k++) {
            float sum = b2[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * w2[j][k];
            }
            outputRaw[k] = sum;
            if (sum > maxLogit) {
                maxLogit = sum;
            }
        }

        float[] output = new float[outputSize];
        float sumExp = 0f;
        for (int k = 0; k < outputSize; k++) {
            output[k] = (float) Math.exp(outputRaw[k] - maxLogit);
            sumExp += output[k];
        }
        for (int k = 0; k < outputSize; k++) {
            output[k] /= sumExp;
        }

        // Cross-entropy loss with label smoothing
        float epsilon = 0.1f; // Label smoothing parameter
        float loss = 0f;
        for (int k = 0; k < outputSize; k++) {
            float smoothedTarget = target[k] * (1 - epsilon) + epsilon / outputSize;
            loss -= smoothedTarget * Math.log(output[k] + 1e-7f);
        }

        // Backpropagation
        float[] dOutput = new float[outputSize];
        for (int k = 0; k < outputSize; k++) {
            float smoothedTarget = target[k] * (1 - epsilon) + epsilon / outputSize;
            dOutput[k] = output[k] - smoothedTarget;
        }

        float[] dHidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            if (dropoutMask[j]) {
                for (int k = 0; k < outputSize; k++) {
                    float grad = dOutput[k];
                    w2[j][k] -= lr * (grad * hidden[j] + regLambda * w2[j][k]);
                    dHidden[j] += grad * w2[j][k];
                }
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            if (dropoutMask[j]) {
                float gradHidden = dHidden[j] * reluDerivative(hiddenRaw[j]);
                for (int i = 0; i < inputSize; i++) {
                    w1[i][j] -= lr * (gradHidden * input[i] + regLambda * w1[i][j]);
                }
                b1[j] -= lr * gradHidden;
            }
        }

        for (int k = 0; k < outputSize; k++) {
            b2[k] -= lr * dOutput[k];
        }

        return loss;
    }

    public PredictionResult predict(float[] input) {
        // Forward pass without dropout
        float[] hidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hidden[j] = relu(sum);
        }

        float[] outputRaw = new float[outputSize];
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < outputSize; k++) {
            float sum = b2[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * w2[j][k];
            }
            outputRaw[k] = sum;
            if (sum > maxLogit) {
                maxLogit = sum;
            }
        }

        float[] output = new float[outputSize];
        float sumExp = 0f;
        for (int k = 0; k < outputSize; k++) {
            output[k] = (float) Math.exp(outputRaw[k] - maxLogit);
            sumExp += output[k];
        }
        for (int k = 0; k < outputSize; k++) {
            output[k] /= sumExp;
        }

        // Calculate entropy to detect uncertainty
        float entropy = 0;
        for (int k = 0; k < outputSize; k++) {
            if (output[k] > 0) {
                entropy -= output[k] * Math.log(output[k]) / Math.log(outputSize);
            }
        }

        int bestIndex = 0;
        float bestVal = output[0];
        for (int k = 1; k < outputSize; k++) {
            if (output[k] > bestVal) {
                bestVal = output[k];
                bestIndex = k;
            }
        }

        // If entropy is high, model is uncertain
        boolean isUncertain = entropy > entropyThreshold;
        return new PredictionResult(bestIndex, bestVal, isUncertain);
    }

    private float relu(float x) {
        return x > 0 ? x : 0.01f * x; // Leaky ReLU
    }

    private float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.01f;
    }
}
