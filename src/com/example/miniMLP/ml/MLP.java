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
    private transient Random rnd = new Random();
    private float relu(float x) {
        return x > 0 ? x : 0;
    }
    private float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        w1 = new float[inputSize][hiddenSize];
        b1 = new float[hiddenSize];
        w2 = new float[hiddenSize][outputSize];
        b2 = new float[outputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                w1[i][j] = rnd.nextFloat() * 2 - 1;
            }
        }
        for (int j = 0; j < hiddenSize; j++) {
            b1[j] = rnd.nextFloat() * 2 - 1;
        }
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                w2[j][k] = rnd.nextFloat() * 2 - 1;
            }
        }
        for (int k = 0; k < outputSize; k++) {
            b2[k] = rnd.nextFloat() * 2 - 1;
        }
    }

    public void train(float[][] inputs, float[][] targets, int epochs, float lr) {
        int n = inputs.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float sumLoss = 0f;
            for (int i = 0; i < n; i++) {
                sumLoss += trainOnExample(inputs[i], targets[i], lr);
            }
            float avgLoss = sumLoss / n;
            System.out.println("Epoch " + epoch + " - Loss: " + avgLoss);
        }
    }

    private float trainOnExample(float[] input, float[] target, float lr) {
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

        float loss = 0f;
        for (int k = 0; k < outputSize; k++) {
            loss -= target[k] * Math.log(output[k] + 1e-7f);
        }

        float[] dOutput = new float[outputSize];
        for (int k = 0; k < outputSize; k++) {
            dOutput[k] = output[k] - target[k];
        }

        float[] dHidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                dHidden[j] += dOutput[k] * w2[j][k];
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                w2[j][k] -= lr * dOutput[k] * hidden[j];
            }
        }
        
        for (int k = 0; k < outputSize; k++) {
            b2[k] -= lr * dOutput[k];
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                w1[i][j] -= lr * dHidden[j] * reluDerivative(hidden[j]) * input[i];
            }
        }
        
        for (int j = 0; j < hiddenSize; j++) {
            b1[j] -= lr * dHidden[j] * reluDerivative(hidden[j]);
        }
        return loss;
    }

    public PredictionResult predict(float[] input) {
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
        int bestIndex = 0;
        float bestVal = output[0];
        for (int k = 1; k < outputSize; k++) {
            if (output[k] > bestVal) {
                bestVal = output[k];
                bestIndex = k;
            }
        }
        return new PredictionResult(bestIndex, bestVal, false, output);
    }
}
