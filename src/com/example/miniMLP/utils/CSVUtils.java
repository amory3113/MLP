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

    // Method to center the image within the grid
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

        // Copy the pixels into the centered position
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                int newY = y - minY + offsetY;
                int newX = x - minX + offsetX;

                if (newY >= 0 && newY < grid && newX >= 0 && newX < grid) {
                    centered[newY][newX] = pixels[y][x];
                }
            }
        }

        return centered;
    }

    /**
     * Обучение MLP на полном наборе данных (без валидации).
     * @param csvFile путь к dataset.csv
     * @param grid размер сетки (56)
     * @return обученная модель MLP
     */
    public static MLP trainMLPFromCSV(String csvFile, int grid) {
        List<float[]> inputList = new ArrayList<>();
        List<float[]> targetList = new ArrayList<>();

        // Считываем весь CSV
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

                // Добавляем аугментацию данных
                addAugmentedData(inVec, targetVec, inputList, targetList, grid);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        if (inputList.isEmpty()) {
            System.err.println("Нет обучающих данных!");
            return null;
        }

        // Конвертируем списки в массивы
        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);

        // Создаём модель и тренируем
        MLP mlp = new MLP(grid * grid, 256, 3);
        mlp.train(inputs, targets, 1250, 0.0005f);

        return mlp;
    }

    // Метод для аугментации
    private static void addAugmentedData(float[] originalInput, float[] originalTarget,
                                         List<float[]> inputList, List<float[]> targetList, int grid) {
        Random rnd = new Random();

        // Преобразуем 1D массив в 2D
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

            float[] augmentedInput = new float[grid * grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    augmentedInput[i * grid + j] = shiftedImage[i][j];
                }
            }

            inputList.add(augmentedInput);
            targetList.add(originalTarget.clone());
        }

        // 2. Добавление шума
        for (int noise = 0; noise < 2; noise++) {
            float[][] noisyImage = new float[grid][grid];
            for (int i = 0; i < grid; i++) {
                for (int j = 0; j < grid; j++) {
                    noisyImage[i][j] = image[i][j];
                    // 5% шанс инвертировать пиксель
                    if (rnd.nextFloat() < 0.05f) {
                        noisyImage[i][j] = 1 - noisyImage[i][j];
                    }
                }
            }

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
        return switch (s) {
            case "e" -> 0;
            case "l" -> 1;
            case "f" -> 2;
            default -> -1;
        };
    }

    /**
     * Тестирование: выводит для каждого рисунка confidence,
     * а в конце среднюю уверенность (вместо точности).
     * Если хотите вывести точность, нужно собрать метки и сравнивать.
     */
    public static float testMLPFromCSV(String csvFile, MLP mlp, int grid) {
        int total = 0;
        float sumConfidence = 0f; // для вычисления средней уверенности

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {
                lineIndex++;
                String[] parts = line.split(",");
                if (parts.length != 1 + grid * grid) {
                    continue;
                }

                // Парсим вход
                float[] inVec = new float[grid * grid];
                for (int i = 0; i < grid * grid; i++) {
                    inVec[i] = Float.parseFloat(parts[i + 1]);
                }

                // Предсказываем
                PredictionResult result = mlp.predict(inVec);

                // Увеличиваем счётчик и суммируем уверенность
                total++;
                sumConfidence += result.confidence;

                // Выводим строку:
                System.out.printf("Рисунок – %d, уверенность – %.2f%%%n", lineIndex, result.confidence * 100);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return 0f;
        }

        if (total == 0) {
            System.out.println("Нет данных для теста (total=0).");
            return 0f;
        }

        // Средняя уверенность
        float avgConfidence = sumConfidence / total;

        // Выводим её
        System.out.printf("Средняя уверенность – %.2f%%%n", avgConfidence * 100);

        // Возвращаем, если требуется
        return avgConfidence;
    }
}
