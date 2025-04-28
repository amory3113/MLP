package com.example.miniMLP.utils;

import com.example.miniMLP.ml.MLP;
import com.example.miniMLP.ml.PredictionResult;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSVUtils {

    public static void savePixelsToCSV(String label, float[][] pix, int grid, String csvFile) {
        float[][] centered = centerImage(pix, grid);
        float[][] binaryPixels = convertToBinary(centered, 0.1f);
        try (FileWriter fw = new FileWriter(csvFile, true)) {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(',');
            for (int y = 0; y < grid; y++)
                for (int x = 0; x < grid; x++) {
                    sb.append((int)binaryPixels[y][x]); // Cast to int to ensure 0 or 1
                    if (!(y == grid - 1 && x == grid - 1)) sb.append(',');
                }
            sb.append('\n');
            fw.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static float[][] convertToBinary(float[][] pixels, float threshold) {
        int height = pixels.length;
        int width = pixels[0].length;
        float[][] binaryPixels = new float[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                binaryPixels[y][x] = pixels[y][x] > threshold ? 1.0f : 0.0f;
            }
        }
        return binaryPixels;
    }

    private static float[][] centerImage(float[][] pix, int grid) {
        int minX = grid, minY = grid, maxX = 0, maxY = 0;
        boolean hasContent = false;
        for (int y = 0; y < grid; y++)
            for (int x = 0; x < grid; x++)
                if (pix[y][x] > 0) {
                    hasContent = true;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
        if (!hasContent)
            return pix;

        float[][] centered = new float[grid][grid];
        int width  = maxX - minX + 1;
        int height = maxY - minY + 1;
        int offX = (grid - width)  / 2;
        int offY = (grid - height) / 2;

        for (int y = minY; y <= maxY; y++)
            for (int x = minX; x <= maxX; x++) {
                int ny = y - minY + offY;
                int nx = x - minX + offX;
                centered[ny][nx] = pix[y][x];
            }
        return centered;
    }

    public static MLP trainMLPFromCSV(String csvFile, int grid) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != 1 + (grid * grid)) {
                    continue;
                }
                String labelStr = parts[0].trim().toLowerCase();

                float[] inVec = new float[grid * grid];
                for (int i = 0; i < grid * grid; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }

                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0 || classIndex >= 3) {
                    continue;
                }
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;

                inputList.add(inVec);
                targetList.add(targetVec);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        if (inputList.isEmpty()) {
            System.err.println("Brak danych szkoleniowych!");
            return null;
        }

        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);
        MLP mlp = new MLP(grid * grid, 128, 3);
        mlp.train(inputs, targets, 600, 0.01f);
        return mlp;
    }

    private static int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        return switch (s) {
            case "e" -> 0;
            case "l" -> 1;
            case "f" -> 2;
            default -> -1;
        };
    }

    public static float testMLPFromCSV(String csvFile, MLP mlp, int grid) {
        int total = 0;
        int correct = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {
                lineIndex++;
                String[] parts = line.split(",");
                if (parts.length != 1 + grid * grid) {
                    continue;
                }

                String trueLabel = parts[0].trim().toLowerCase();
                int trueIndex = symbolToIndex(trueLabel);
                if (trueIndex < 0) {
                    System.out.println("Nieprawidłowa etykieta w linii " + lineIndex + ": " + trueLabel);
                    continue;
                }

                float[] inVec = new float[grid * grid];
                for (int i = 0; i < grid * grid; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }
                PredictionResult result = mlp.predict(inVec);
                total++;
                if (result.predictedIndex == trueIndex) {
                    correct++;
                }

                String predictedLabel = indexToSymbol(result.predictedIndex);
                System.out.printf("Rysunek %d: prawdziwy=%s, przewidywany=%s, pewność=%.2f%n", lineIndex, trueLabel, predictedLabel, result.confidence);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return 0f;
        }

        if (total == 0) {
            System.out.println("Brak danych dla testu.");
            return 0f;
        }

        float accuracy = (float) correct / total;
        System.out.printf("Razem testów: %d, Poprawnych: %d%n", total, correct);
        System.out.printf("Dokładność: %.2f%n", accuracy);
        return accuracy;
    }
    
    private static String indexToSymbol(int idx) {
        switch (idx) {
            case 0: return "e";
            case 1: return "l";
            case 2: return "f";
            default: return "?";
        }
    }
}
