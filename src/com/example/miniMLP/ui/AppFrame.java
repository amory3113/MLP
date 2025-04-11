package com.example.miniMLP.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;

import com.example.miniMLP.ml.MLP;
import com.example.miniMLP.ml.PredictionResult;
import com.example.miniMLP.utils.CSVUtils;
import com.example.miniMLP.utils.ModelUtils;

public class AppFrame extends JFrame {
    private static final int WIDTH = 1440;
    private static final int HEIGHT = 750;

    private final int GRID = 56;

    private BufferedImage canvas;
    private Graphics2D g2;
    private int[][] binaryPixels = new int[GRID][GRID];

    private JRadioButton eRadio;
    private JRadioButton lRadio;
    private JRadioButton fRadio;

    private MLP mlpModel;

    private DrawingPanel drawingPanel;

    public AppFrame() {
        super("MLP");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());

        canvas = new BufferedImage(WIDTH / 2, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);

        drawingPanel = new DrawingPanel(canvas);
        add(drawingPanel, BorderLayout.CENTER);

        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        Font btnFont = new Font("Times new roman", Font.BOLD, 40);
        Font radioFont = new Font("Times new roman", Font.BOLD, 50);

        JPanel row1Panel = new JPanel(new BorderLayout());
        JPanel row1Left = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row1Left.setBorder(BorderFactory.createEmptyBorder(0, 25, 0, 0));
        JButton clearBtn = new JButton("czyscic");
        clearBtn.setFont(btnFont);
        clearBtn.setPreferredSize(new Dimension(300, 100));
        clearBtn.addActionListener(e -> clearCanvas(drawingPanel));
        row1Left.add(clearBtn);

        JPanel row1Right = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        row1Right.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 25));
        JButton recognizeBtn = new JButton("rozpoznac");
        recognizeBtn.setFont(btnFont);
        recognizeBtn.addActionListener(e -> recognizeSymbol());
        recognizeBtn.setPreferredSize(new Dimension(300, 100));
        row1Right.add(recognizeBtn);
        row1Panel.add(row1Left, BorderLayout.WEST);
        row1Panel.add(row1Right, BorderLayout.EAST);
        row1Panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, clearBtn.getPreferredSize().height + 10));

        JPanel row2Panel = new JPanel(new BorderLayout());
        JPanel row2Left = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row2Left.setBorder(BorderFactory.createEmptyBorder(0, 25, 0, 0));
        JButton saveBtn = new JButton("zapisac");
        saveBtn.setFont(btnFont);
        saveBtn.setPreferredSize(new Dimension(300, 100));
        saveBtn.addActionListener(e -> {
            saveToCSV();
            clearCanvas(drawingPanel);
        });
        row2Left.add(saveBtn);

        JPanel row2Right = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        row2Right.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 25));
        JButton trainBtn = new JButton("ucz MLP");
        trainBtn.setFont(btnFont);
        trainBtn.addActionListener(e -> trainModel());
        trainBtn.setPreferredSize(new Dimension(300, 100));
        row2Right.add(trainBtn);
        row2Panel.add(row2Left, BorderLayout.WEST);
        row2Panel.add(row2Right, BorderLayout.EAST);
        row2Panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, saveBtn.getPreferredSize().height + 10));

        JPanel row3Panel = new JPanel(new BorderLayout());
        row3Panel.setBorder(BorderFactory.createEmptyBorder(25, 25, 0, 25));
        JButton testBtn = new JButton("testuj");
        testBtn.setFont(btnFont);
        testBtn.addActionListener(e -> testAction());
        testBtn.setPreferredSize(new Dimension(610, 100));
        JPanel buttonWrapper = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 0));
        buttonWrapper.add(testBtn);
        row3Panel.add(buttonWrapper, BorderLayout.CENTER);
        row3Panel.setMaximumSize(row3Panel.getPreferredSize());

        eRadio = new JRadioButton("e", true);
        eRadio.setFont(radioFont);
        lRadio = new JRadioButton("l");
        lRadio.setFont(radioFont);
        fRadio = new JRadioButton("f");
        fRadio.setFont(radioFont);
        ButtonGroup group = new ButtonGroup();
        group.add(eRadio);
        group.add(lRadio);
        group.add(fRadio);

        JPanel radioPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 150, 0));
        radioPanel.setBorder(BorderFactory.createEmptyBorder(25, 25, 0, 25));
        radioPanel.add(eRadio);
        radioPanel.add(lRadio);
        radioPanel.add(fRadio);

        rightPanel.add(row1Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row2Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row3Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(radioPanel);

        add(rightPanel, BorderLayout.EAST);

        String modelPath = "mlpModel.bin";
        if (new File(modelPath).exists()) {
            mlpModel = ModelUtils.loadModel(modelPath);
        }

        setVisible(true);
    }

    private void clearCanvas(JPanel drawingPanel) {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    private void recognizeSymbol() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this, "Najpierw wytrenuj model lub pobierz go!");
            return;
        }
        readPixelsFromCanvas();

        // Center the image before recognition
        int[][] centeredPixels = centerImage(binaryPixels, GRID);

        // Проверка на пустой рисунок
        if (isEmptyDrawing(centeredPixels)) {
            JOptionPane.showMessageDialog(this, "Нарисуйте что-то для распознавания!");
            return;
        }

        float[] inputVec = convertToFloatVector(centeredPixels);
        PredictionResult result = mlpModel.predict(inputVec);
        String symbol = indexToSymbol(result.predictedIndex);

        // Проверяем неопределенность и порог уверенности
        if (result.isUncertain || result.confidence < 0.7f) {
            JOptionPane.showMessageDialog(this, "Символ не распознан! Система неуверена в предсказании.");
        } else {
            String confidenceText = String.format("%.2f%%", result.confidence * 100);
            JOptionPane.showMessageDialog(this, "MLP уверена, что это: " + symbol +
                    " (вероятность: " + confidenceText + ")");
        }
    }

    // Метод для определения пустого рисунка
    private boolean isEmptyDrawing(int[][] pixels) {
        int pixelCount = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                if (pixels[y][x] > 0) {
                    pixelCount++;
                }
            }
        }

        // Если меньше 1% пикселей заполнено, считаем рисунок пустым
        return pixelCount < (GRID * GRID * 0.01);
    }


    private void saveToCSV() {
        readPixelsFromCanvas();
        String label = getSelectedLabel();
//        if (label == null) {
//            JOptionPane.showMessageDialog(this, "Nie wybrano żadnej litery!");
//            return;
//        }
        CSVUtils.savePixelsToCSV(label, binaryPixels, GRID, "dataset.csv");
        //JOptionPane.showMessageDialog(this, "Zapisano jako '" + label + "'");
    }

    private void trainModel() {
        mlpModel = CSVUtils.trainMLPFromCSV("dataset.csv", GRID);
        if (mlpModel != null) {
            ModelUtils.saveModel(mlpModel, "mlpModel.bin");
            JOptionPane.showMessageDialog(this, "Model został pomyślnie wytrenowany i zapisany!");
        }
    }

    private void testAction() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this,
                    "Najpierw wytrenuj model (ucz MLP) lub załaduj go!");
            return;
        }

        String testCsvFile = "dataset_test.csv";

        File f = new File(testCsvFile);
        if (!f.exists()) {
            JOptionPane.showMessageDialog(this,
                    "Plik testowy " + testCsvFile + " nie istnieje!");
            return;
        }

        float accuracy = CSVUtils.testMLPFromCSV(testCsvFile, mlpModel, GRID);

        JOptionPane.showMessageDialog(this,
                "Dokładność (accuracy) na zbiorze testowym: " + (accuracy * 100f) + " %");
    }

    private void readPixelsFromCanvas() {
        int cellSize = canvas.getWidth() / GRID;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                int blackCount = 0;
                for (int dy = 0; dy < cellSize; dy++) {
                    for (int dx = 0; dx < cellSize; dx++) {
                        int px = x * cellSize + dx;
                        int py = y * cellSize + dy;
                        int color = canvas.getRGB(px, py) & 0xFF;
                        if (color < 128) {
                            blackCount++;
                        }
                    }
                }
                double ratio = blackCount / (double) (cellSize * cellSize);
                binaryPixels[y][x] = (ratio > 0.2) ? 1 : 0;
            }
        }
    }

    // Method to center the image within the grid
    private int[][] centerImage(int[][] pixels, int grid) {
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

    private float[] convertToFloatVector(int[][] arr) {
        float[] vec = new float[GRID * GRID];
        int index = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                vec[index++] = arr[y][x];
            }
        }
        return vec;
    }

    private String getSelectedLabel() {
        if (eRadio.isSelected()) return "e";
        if (lRadio.isSelected()) return "l";
        if (fRadio.isSelected()) return "f";
        return null;
    }

    private String indexToSymbol(int idx) {
        switch (idx) {
            case 0:
                return "e";
            case 1:
                return "l";
            case 2:
                return "f";
            default:
                return "?";
        }
    }
}