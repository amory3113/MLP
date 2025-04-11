package com.example.miniMLP.utils;

import com.example.miniMLP.ml.MLP;
import com.example.miniMLP.ml.PredictionResult;
import java.util.Random;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSVUtils {
    public static void savePixelsToCSV(String label, int[][] pixels, int grid, String csvFile) {
        // Center the image before saving
        int[][] centeredPixels = centerImage(pixels, grid);

        try (FileWriter fw = new FileWriter(csvFile, true)) {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(",");
            for (int y = 0; y < grid; y++) {
                for (int x = 0; x < grid; x++) {
                    sb.append(centeredPixels[y][x]);
                    if (!(y == grid - 1 && x == grid - 1)) {
                        sb.append(",");
                    }
                }
            }
            sb.append("\n");
            fw.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // New method to center the image within the grid
    private static int[][] centerImage(int[][] pixels, int grid) {
        // Find the bounding box of the drawing
        int minX = grid;
        int minY = grid;
        int maxX = 0;
        int maxY = 0;
        boolean hasContent = false;

        // Find boundaries of the drawing
        for (int y = 0; y < grid; y++) {
            for (int x = 0; x < grid; x++) {
                if (pixels[y][x] > 0) {
                    hasContent = true;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        // If no drawing found, return original
        if (!hasContent) {
            return pixels;
        }

        // Create a new grid with the centered image
        int[][] centered = new int[grid][grid];

        // Calculate width and height of drawing
        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        // Calculate offsets to center
        int offsetX = (grid - width) / 2;
        int offsetY = (grid - height) / 2;

        // Copy the pixels to the centered position
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                int newY = y - minY + offsetY;
                int newX = x - minX + offsetX;

                // Make sure we don't go out of bounds
                if (newY >= 0 && newY < grid && newX >= 0 && newX < grid) {
                    centered[newY][newX] = pixels[y][x];
                }
            }
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
                if (parts.length != 1 + (grid * grid)) continue;
                String labelStr = parts[0].trim();
                float[] inVec = new float[grid * grid];
                for (int i = 0; i < grid * grid; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }
                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0 || classIndex >= 3) continue;
                float[] targetVec = new float[3];
                targetVec[classIndex] = 1.0f;
                inputList.add(inVec);
                targetList.add(targetVec);

                // Добавляем аугментацию данных - небольшие смещения и повороты
                addAugmentedData(inVec, targetVec, inputList, targetList, grid);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        if (inputList.isEmpty()) {
            System.err.println("Brak danych szkoleniowych!");
            return null;
        }

        // Разделение на обучающую и валидационную выборки
        int trainSize = (int) (inputList.size() * 0.8);
        List<float[]> trainInputs = new ArrayList<>();
        List<float[]> trainTargets = new ArrayList<>();
        List<float[]> valInputs = new ArrayList<>();
        List<float[]> valTargets = new ArrayList<>();

        // Перемешиваем данные
        Random rnd = new Random(42);
        for (int i = 0; i < inputList.size(); i++) {
            int j = rnd.nextInt(inputList.size());

            float[] tempInput = inputList.get(i);
            inputList.set(i, inputList.get(j));
            inputList.set(j, tempInput);

            float[] tempTarget = targetList.get(i);
            targetList.set(i, targetList.get(j));
            targetList.set(j, tempTarget);
        }

        // Разделяем на обучающую и валидационную выборки
        for (int i = 0; i < inputList.size(); i++) {
            if (i < trainSize) {
                trainInputs.add(inputList.get(i));
                trainTargets.add(targetList.get(i));
            } else {
                valInputs.add(inputList.get(i));
                valTargets.add(targetList.get(i));
            }
        }

        float[][] trainInputsArray = trainInputs.toArray(new float[0][]);
        float[][] trainTargetsArray = trainTargets.toArray(new float[0][]);

        MLP mlp = new MLP(grid * grid, 256, 3); // Увеличиваем скрытый слой для лучшей емкости
        mlp.train(trainInputsArray, trainTargetsArray, 1250, 0.0005f);

        // Проверяем точность на валидационной выборке
        int correct = 0;
        for (int i = 0; i < valInputs.size(); i++) {
            PredictionResult result = mlp.predict(valInputs.get(i));
            int trueClass = 0;
            for (int j = 0; j < 3; j++) {
                if (valTargets.get(i)[j] > 0.5) {
                    trueClass = j;
                    break;
                }
            }

            if (result.predictedIndex == trueClass && !result.isUncertain) {
                correct++;
            }
        }

        float valAccuracy = (float) correct / valInputs.size();
        System.out.println("Валидационная точность: " + (valAccuracy * 100) + "%");

        return mlp;
    }

    // Метод для аугментации данных
    private static void addAugmentedData(float[] originalInput, float[] originalTarget,
                                         List<float[]> inputList, List<float[]> targetList, int grid) {
        Random rnd = new Random();

        // Преобразуем линейный массив в 2D для удобства манипуляций
        float[][] image = new float[grid][grid];
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                image[i][j] = originalInput[i * grid + j];
            }
        }

        // 1. Небольшой сдвиг (±2 пикселя)
        for (int shift = 0; shift < 3; shift++) {
            int dx = rnd.nextInt(5) - 2;
            int dy = rnd.nextInt(5) - 2;

            float[][] shiftedImage = new float[grid][grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    int newI = i + dy;
                    int newJ = j + dx;
                    if (newI >= 0 && newI < grid && newJ >= 0 && newJ < grid) {
                        shiftedImage[i][j] = image[newI][newJ];
                    }
                }
            }

            // Добавляем в набор данных
            float[] augmentedInput = new float[grid * grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    augmentedInput[i * grid + j] = shiftedImage[i][j];
                }
            }

            inputList.add(augmentedInput);
            targetList.add(originalTarget.clone());
        }

        // 2. Добавление небольшого шума
        for (int noise = 0; noise < 2; noise++) {
            float[][] noisyImage = new float[grid][grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    noisyImage[i][j] = image[i][j];

                    // 5% шанс инвертировать пиксель
                    if (rnd.nextFloat() < 0.05) {
                        noisyImage[i][j] = 1 - noisyImage[i][j];
                    }
                }
            }

            // Добавляем в набор данных
            float[] augmentedInput = new float[grid * grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    augmentedInput[i * grid + j] = noisyImage[i][j];
                }
            }

            inputList.add(augmentedInput);
            targetList.add(originalTarget.clone());
        }
    }

    private static int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        if (s.equals("e")) return 0;
        else if (s.equals("l")) return 1;
        else if (s.equals("f")) return 2;
        return -1;
    }

    public static float testMLPFromCSV(String csvFile, MLP mlp, int grid) {
        int correct = 0;
        int total = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length != 1 + grid * grid) {
                    continue;
                }

                String labelStr = parts[0].trim().toLowerCase();
                int classIndex = symbolToIndex(labelStr);
                if (classIndex < 0) {
                    continue;
                }

                float[] inVec = new float[grid * grid];
                for (int i = 0; i < grid * grid; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }

                PredictionResult result = mlp.predict(inVec);

                if (result.predictedIndex == classIndex) {
                    correct++;
                }
                total++;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return 0f;
        }

        if (total == 0) {
            return 0f;
        }
        return correct / (float) total;
    }
}