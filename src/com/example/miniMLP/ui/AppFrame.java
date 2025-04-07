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

    // Чтобы можно было очищать холст из других методов, сохраним ссылку на панель:
    private DrawingPanel drawingPanel;

    public AppFrame() {
        super("MLP");
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());

        // Инициализация холста (левая часть)
        canvas = new BufferedImage(WIDTH / 2, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);

        drawingPanel = new DrawingPanel(canvas);
        add(drawingPanel, BorderLayout.CENTER);

        // Правая часть: единая вертикальная панель
        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        Font btnFont = new Font("Times new roman", Font.BOLD, 40);
        Font radioFont = new Font("Times new roman", Font.BOLD, 50);

        // ----- Первая строка: "czyscic" слева и "rozpoznac" справа -----
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

        // ----- Вторая строка: "zapisac" слева и "ucz MLP" справа -----
        JPanel row2Panel = new JPanel(new BorderLayout());
        JPanel row2Left = new JPanel(new FlowLayout(FlowLayout.LEFT));
        row2Left.setBorder(BorderFactory.createEmptyBorder(0, 25, 0, 0));
        JButton saveBtn = new JButton("zapisac");
        saveBtn.setFont(btnFont);
        saveBtn.setPreferredSize(new Dimension(300, 100));
        // При сохранении теперь автоматически очищаем холст
        saveBtn.addActionListener(e -> {
            saveToCSV();
            clearCanvas(drawingPanel); // <-- добавлено
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

        // ----- Третья строка: большая кнопка "testuj" -----
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

        // ----- Четвёртая строка: панель с радио-кнопками -----
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

        // Добавляем все строки в правую панель
        rightPanel.add(row1Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row2Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(row3Panel);
        rightPanel.add(Box.createVerticalStrut(10));
        rightPanel.add(radioPanel);

        add(rightPanel, BorderLayout.EAST);

        // Если модель существует, загружаем
        String modelPath = "mlpModel.bin";
        if (new File(modelPath).exists()) {
            mlpModel = ModelUtils.loadModel(modelPath);
        }

        setVisible(true);
    }

    // ---- Метод очистки холста (не трогаем логику)
    private void clearCanvas(JPanel drawingPanel) {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    // ---- При нажатии "rozpoznac"
    private void recognizeSymbol() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this, "Сначала обучите модель или загрузите её!");
            return;
        }
        readPixelsFromCanvas();
        float[] inputVec = convertToFloatVector(binaryPixels);
        PredictionResult result = mlpModel.predict(inputVec);
        String symbol = indexToSymbol(result.predictedIndex);
        if (result.confidence < 0.005f) {
            JOptionPane.showMessageDialog(this, "Символ не распознан, уверенность слишком низкая!");
        } else {
            JOptionPane.showMessageDialog(this, "MLP думает, что это: " + symbol +
                    " (вероятность: " + result.confidence + ")");
        }
    }

    // ---- При нажатии "zapisac"
    private void saveToCSV() {
        readPixelsFromCanvas();
        String label = getSelectedLabel();
        if (label == null) {
            JOptionPane.showMessageDialog(this, "Не выбрана буква!");
            return;
        }
        CSVUtils.savePixelsToCSV(label, binaryPixels, GRID, "dataset.csv");
        JOptionPane.showMessageDialog(this, "Сохранено как '" + label + "'");
    }

    // ---- При нажатии "ucz MLP"
    private void trainModel() {
        mlpModel = CSVUtils.trainMLPFromCSV("dataset.csv", GRID);
        if (mlpModel != null) {
            ModelUtils.saveModel(mlpModel, "mlpModel.bin");
            JOptionPane.showMessageDialog(this, "Модель успешно обучена и сохранена!");
        }
    }

    // ---- При нажатии "testuj"
    private void testAction() {
        JOptionPane.showMessageDialog(this, "Нажата кнопка 'testuj'!");
    }

    // ---- Читаем пиксели
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
