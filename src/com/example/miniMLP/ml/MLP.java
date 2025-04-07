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

    // Random не сериализуем (transient), чтобы не мешать сохранению/загрузке модели:
    private transient Random rnd = new Random();

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
        // Инициализируем веса небольшими случайными значениями
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = (rnd.nextFloat() - 0.5f) * 0.2f;
            }
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

    /**
     * Тренировка на одном примере (forward + backward).
     * Возвращает loss (кросс-энтропия).
     */
    private float trainOnExample(float[] input, float[] target, float lr) {
        // ========== Forward pass ==========

        // 1) Скрытый слой (hiddenRaw, затем применяем ReLU)
        float[] hiddenRaw = new float[hiddenSize];
        float[] hidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hiddenRaw[j] = sum;
            hidden[j] = relu(sum);  // Leaky ReLU 0.01
        }

        // 2) Выходной слой (outputRaw), потом softmax
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

        // Softmax (чтобы избежать переполнения, вычитаем maxLogit)
        float[] output = new float[outputSize];
        float sumExp = 0f;
        for (int k = 0; k < outputSize; k++) {
            output[k] = (float) Math.exp(outputRaw[k] - maxLogit);
            sumExp += output[k];
        }
        for (int k = 0; k < outputSize; k++) {
            output[k] /= sumExp;
        }

        // 3) Функция потерь (кросс-энтропия)
        float loss = 0f;
        for (int k = 0; k < outputSize; k++) {
            // - sum( y_true * log(y_pred) )
            loss -= target[k] * Math.log(output[k] + 1e-7f);
        }

        // ========== Backward pass ==========

        // dOutput = (softmax - target)
        float[] dOutput = new float[outputSize];
        for (int k = 0; k < outputSize; k++) {
            dOutput[k] = output[k] - target[k];
        }

        // Распространяем ошибку обратно на w2, b2, и считаем dHidden
        float[] dHidden = new float[hiddenSize];
        for (int k = 0; k < outputSize; k++) {
            float grad = dOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                w2[j][k] -= lr * grad * hidden[j];
                dHidden[j] += grad * w2[j][k];
            }
            b2[k] -= lr * grad;
        }

        // Учитываем производную ReLU
        for (int j = 0; j < hiddenSize; j++) {
            // Если hiddenRaw[j] <= 0, используем 0.01
            // (т.е. leaky ReLU, как в relu(...))
            if (hiddenRaw[j] <= 0) {
                dHidden[j] *= 0.01f;
            }
        }

        // dHidden распространяем на w1, b1
        for (int j = 0; j < hiddenSize; j++) {
            float grad = dHidden[j];
            for (int i = 0; i < inputSize; i++) {
                w1[i][j] -= lr * grad * input[i];
            }
            b1[j] -= lr * grad;
        }

        return loss;
    }

    /**
     * Предсказание (forward pass + argmax).
     */
    public PredictionResult predict(float[] input) {
        // ========== Forward pass (без backprop) ==========

        // 1) Скрытый слой
        float[] hidden = new float[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            float sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            hidden[j] = relu(sum);
        }

        // 2) Выходной слой (raw), затем softmax
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

        // Ищем индекс максимального значения (argmax)
        int bestIndex = 0;
        float bestVal = output[0];
        for (int k = 1; k < outputSize; k++) {
            if (output[k] > bestVal) {
                bestVal = output[k];
                bestIndex = k;
            }
        }

        return new PredictionResult(bestIndex, bestVal);
    }

    /**
     * Leaky ReLU (0.01) — если нужно,
     * можно сделать классическую ReLU: return (x > 0) ? x : 0f;
     */
    private float relu(float x) {
        return (x > 0) ? x : 0.01f * x;
    }
}
