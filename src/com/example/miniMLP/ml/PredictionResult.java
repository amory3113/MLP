package com.example.miniMLP.ml;

public class PredictionResult {
    public int predictedIndex;
    public float confidence;
    public boolean isUncertain; // Новое поле для определения неуверенности

    public PredictionResult(int predictedIndex, float confidence) {
        this.predictedIndex = predictedIndex;
        this.confidence = confidence;
        this.isUncertain = false;
    }

    public PredictionResult(int predictedIndex, float confidence, boolean isUncertain) {
        this.predictedIndex = predictedIndex;
        this.confidence = confidence;
        this.isUncertain = isUncertain;
    }
}