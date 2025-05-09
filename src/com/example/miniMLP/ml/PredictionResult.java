package com.example.miniMLP.ml;

public class PredictionResult {
    public int predictedIndex;
    public float confidence;
    public boolean isUncertain;
    public float[] probabilities;

    public PredictionResult(int predictedIndex, float confidence, boolean isUncertain, float[] probabilities) {
        this.predictedIndex = predictedIndex;
        this.confidence = confidence;
        this.isUncertain = isUncertain;
        this.probabilities = probabilities;
    }
}
