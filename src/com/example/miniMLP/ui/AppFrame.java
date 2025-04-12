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

        // Инициализируем холст
        canvas = new BufferedImage(WIDTH / 2, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);

        drawingPanel = new DrawingPanel(canvas);
        add(drawingPanel, BorderLayout.CENTER);

        // Правая панель (панели с кнопками)
        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        Font btnFont = new Font("Times new roman", Font.BOLD, 40);
        Font radioFont = new Font("Times new roman", Font.BOLD, 50);

        // ----------- Row1: "czyscic" + "rozpoznac" -----------
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
        recognizeBtn.setPreferredSize(new Dimension(300, 100));
        recognizeBtn.addActionListener(e -> recognizeSymbol());
        row1Right.add(recognizeBtn);

        row1Panel.add(row1Left, BorderLayout.WEST);
        row1Panel.add(row1Right, BorderLayout.EAST);
        row1Panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, clearBtn.getPreferredSize().height + 10));

        // ----------- Row2: "zapisac" + "ucz MLP" -----------
        JPanel row2Panel = new JPanel(new BorderLayout());
        JPanel row2Left = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row2Left.setBorder(BorderFactory.createEmptyBorder(0, 25, 0, 0));
        JButton saveBtn = new JButton("zapisac");
        saveBtn.setFont(btnFont);
        saveBtn.setPreferredSize(new Dimension(300, 100));
        saveBtn.addActionListener(e -> {
            // Сохраняем текущий рисунок в dataset.csv
            saveToCSV("dataset.csv");
            clearCanvas(drawingPanel);
        });
        row2Left.add(saveBtn);

        JPanel row2Right = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        row2Right.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 25));
        JButton trainBtn = new JButton("ucz MLP");
        trainBtn.setFont(btnFont);
        trainBtn.setPreferredSize(new Dimension(300, 100));
        trainBtn.addActionListener(e -> trainModel());
        row2Right.add(trainBtn);

        row2Panel.add(row2Left, BorderLayout.WEST);
        row2Panel.add(row2Right, BorderLayout.EAST);
        row2Panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, saveBtn.getPreferredSize().height + 10));

        // ----------- Row3: "Zapisz test" + "testuj" -----------
        JPanel row3Panel = new JPanel(new BorderLayout());
        row3Panel.setBorder(BorderFactory.createEmptyBorder(25, 25, 0, 25));

        // Левая кнопка: Сохранение в dataset_test.csv
        JPanel row3Left = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton saveTestBtn = new JButton("Zapisz test");
        saveTestBtn.setFont(btnFont);
        saveTestBtn.setPreferredSize(new Dimension(300, 100));
        saveTestBtn.addActionListener(e -> {
            // Сохраняем в dataset_test.csv
            saveToCSV("dataset_test.csv");
            clearCanvas(drawingPanel);
        });
        row3Left.add(saveTestBtn);

        // Правая кнопка: testuj
        JPanel row3Right = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton testBtn = new JButton("testuj");
        testBtn.setFont(btnFont);
        testBtn.setPreferredSize(new Dimension(300, 100));
        testBtn.addActionListener(e -> testAction());
        row3Right.add(testBtn);

        row3Panel.add(row3Left, BorderLayout.WEST);
        row3Panel.add(row3Right, BorderLayout.EAST);
        row3Panel.setMaximumSize(new Dimension(Integer.MAX_VALUE, saveTestBtn.getPreferredSize().height + 10));

        // ----------- Радиокнопки e / l / f -----------
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

        // Добавляем все row-панели на правую панель
        rightPanel.add(row1Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row2Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row3Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(radioPanel);

        add(rightPanel, BorderLayout.EAST);

        // Если модель уже есть на диске, загрузим её
        String modelPath = "mlpModel.bin";
        if (new File(modelPath).exists()) {
            mlpModel = ModelUtils.loadModel(modelPath);
        }

        setVisible(true);
    }

    // Очищаем холст
    private void clearCanvas(JPanel drawingPanel) {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    // Распознаём нарисованный символ
    private void recognizeSymbol() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this, "Najpierw wytrenuj model lub pobierz go!");
            return;
        }
        readPixelsFromCanvas();

        int[][] centeredPixels = centerImage(binaryPixels, GRID);
        if (isEmptyDrawing(centeredPixels)) {
            JOptionPane.showMessageDialog(this, "Нарисуйте что-то для распознавания!");
            return;
        }

        float[] inputVec = convertToFloatVector(centeredPixels);
        PredictionResult result = mlpModel.predict(inputVec);
        String symbol = indexToSymbol(result.predictedIndex);

        if (result.isUncertain || result.confidence < 0.7f) {
            JOptionPane.showMessageDialog(this, "Символ не распознан! Система неуверена в предсказании.");
        } else {
            String confidenceText = String.format("%.2f%%", result.confidence * 100);
            JOptionPane.showMessageDialog(this, "MLP уверена, что это: " + symbol +
                    " (вероятность: " + confidenceText + ")");
        }
    }

    // Сохраняем текущий рисунок с выбранной меткой в указанный CSV (dataset.csv или dataset_test.csv)
    private void saveToCSV(String csvFile) {
        readPixelsFromCanvas();
        String label = getSelectedLabel();
        if (label == null) {
            JOptionPane.showMessageDialog(this, "Не выбрана буква (e/l/f)!");
            return;
        }
        CSVUtils.savePixelsToCSV(label, binaryPixels, GRID, csvFile);
        // Можно вывести сообщение, что сохранилось
        // JOptionPane.showMessageDialog(this, "Сохранено в файл: " + csvFile);
    }

    // Обучение модели (использует dataset.csv)
    private void trainModel() {
        mlpModel = CSVUtils.trainMLPFromCSV("dataset.csv", GRID);
        if (mlpModel != null) {
            ModelUtils.saveModel(mlpModel, "mlpModel.bin");
            JOptionPane.showMessageDialog(this, "Модель успешно обучена и сохранена (mlpModel.bin)!");
        }
    }

    // Тестируем модель на файле dataset_test.csv
    private void testAction() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this,
                    "Сначала обучите модель (ucz MLP) или загрузите её!");
            return;
        }

        String testCsvFile = "dataset_test.csv";
        File f = new File(testCsvFile);
        if (!f.exists()) {
            JOptionPane.showMessageDialog(this,
                    "Файл для теста не найден: " + testCsvFile);
            return;
        }

        float accuracy = CSVUtils.testMLPFromCSV(testCsvFile, mlpModel, GRID);
        float percentage = accuracy * 100f;
        JOptionPane.showMessageDialog(this,
                "Точность на тестовом наборе (" + testCsvFile + "): " + percentage + " %");
    }

    // Проверяем, не пустой ли рисунок
    private boolean isEmptyDrawing(int[][] pixels) {
        int pixelCount = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                if (pixels[y][x] > 0) {
                    pixelCount++;
                }
            }
        }
        return pixelCount < (GRID * GRID * 0.01);
    }

    // Читаем пиксели с холста в массив binaryPixels
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

    // Центрирование изображения
    private int[][] centerImage(int[][] pixels, int grid) {
        int minX = grid, minY = grid, maxX = 0, maxY = 0;
        boolean hasContent = false;

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
        if (!hasContent) {
            return pixels; // Пустое изображение
        }

        int[][] centered = new int[grid][grid];
        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        int offsetX = (grid - width) / 2;
        int offsetY = (grid - height) / 2;

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

    // Конвертация 2D массива в float-вектор
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

    // Определяем, какая буква выбрана
    private String getSelectedLabel() {
        if (eRadio.isSelected()) return "e";
        if (lRadio.isSelected()) return "l";
        if (fRadio.isSelected()) return "f";
        return null;
    }

    // Индекс -> буква
    private String indexToSymbol(int idx) {
        switch (idx) {
            case 0: return "e";
            case 1: return "l";
            case 2: return "f";
            default: return "?";
        }
    }
}
