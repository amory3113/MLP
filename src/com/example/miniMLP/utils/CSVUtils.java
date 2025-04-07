package com.example.miniMLP.utils;

import com.example.miniMLP.ml.MLP;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSVUtils {
    public static void savePixelsToCSV(String label, int[][] pixels, int grid, String csvFile) {
        try (FileWriter fw = new FileWriter(csvFile, true)) {
            StringBuilder sb = new StringBuilder();
            sb.append(label).append(",");
            for (int y = 0; y < grid; y++) {
                for (int x = 0; x < grid; x++) {
                    sb.append(pixels[y][x]);
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
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        if (inputList.isEmpty()) {
            System.err.println("Немає даних для тренування!");
            return null;
        }
        float[][] inputs = inputList.toArray(new float[0][]);
        float[][] targets = targetList.toArray(new float[0][]);
        MLP mlp = new MLP(grid * grid, 256, 3);
        mlp.train(inputs, targets, 600, 0.001f);
        return mlp;
    }

    private static int symbolToIndex(String s) {
        s = s.trim().toLowerCase();
        if (s.equals("e")) return 0;
        else if (s.equals("l")) return 1;
        else if (s.equals("f")) return 2;
        return -1;
    }
}

