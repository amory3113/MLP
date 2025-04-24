package com.example.miniMLP.ui;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;

import com.example.miniMLP.ml.MLP;
import com.example.miniMLP.ml.PredictionResult;
import com.example.miniMLP.utils.CSVUtils;
import com.example.miniMLP.utils.ModelUtils;

public class AppFrame extends JFrame {
    private static final int WIDTH = 1200;
    private static final int HEIGHT = 750;
    private static final int DRAW_PANEL_WIDTH = 600;

    private final int GRID = 28;

    private BufferedImage canvas;
    private Graphics2D g2;
    private float[][] pixels = new float[GRID][GRID];

    private JRadioButton eRadio;
    private JRadioButton lRadio;
    private JRadioButton fRadio;

    private MLP mlpModel;

    private DrawingPanel drawingPanel;

    public AppFrame() {
        super("MLP Character Recognition");
        setupLookAndFeel();
        
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(10, 10));
        
        // Set background color for the main frame
        getContentPane().setBackground(new Color(240, 240, 245));

        // Create drawing panel on the left side
        canvas = new BufferedImage(DRAW_PANEL_WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2 = canvas.createGraphics();
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);

        drawingPanel = new DrawingPanel(canvas);
        drawingPanel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createEmptyBorder(10, 10, 10, 5),
            BorderFactory.createLineBorder(new Color(150, 150, 150), 1)
        ));
        
        JPanel leftPanel = new JPanel(new BorderLayout());
        leftPanel.setOpaque(false);
        leftPanel.add(drawingPanel, BorderLayout.CENTER);
        
        add(leftPanel, BorderLayout.WEST);

        // Create right panel with controls
        JPanel rightPanel = createControlPanel();
        add(rightPanel, BorderLayout.CENTER);

        // Load model if exists
        String modelPath = "mlpModel.bin";
        if (new File(modelPath).exists()) {
            mlpModel = ModelUtils.loadModel(modelPath);
        }

        setVisible(true);
    }
    
    private void setupLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            System.out.println("Could not set system look and feel: " + e);
        }
    }

    private JPanel createControlPanel() {
        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
        controlPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        controlPanel.setOpaque(false);

        // Common button style
        Font buttonFont = new Font("Arial", Font.BOLD, 18);
        Dimension buttonSize = new Dimension(250, 80);
        
        // Create action buttons panel
        JPanel actionsPanel = new JPanel();
        actionsPanel.setOpaque(false);
        actionsPanel.setLayout(new GridLayout(3, 2, 30, 30));
        actionsPanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 20, 0));
        
        // Clear Button
        JButton clearBtn = createStyledButton("Clear", buttonFont);
        clearBtn.setPreferredSize(buttonSize);
        clearBtn.addActionListener(e -> clearCanvas(drawingPanel));
        actionsPanel.add(clearBtn);
        
        // Recognize Button
        JButton recognizeBtn = createStyledButton("Recognize", buttonFont);
        recognizeBtn.setPreferredSize(buttonSize);
        recognizeBtn.addActionListener(e -> recognizeSymbol());
        actionsPanel.add(recognizeBtn);
        
        // Save Button
        JButton saveBtn = createStyledButton("Save to Dataset", buttonFont);
        saveBtn.setPreferredSize(buttonSize);
        saveBtn.addActionListener(e -> {
            saveToCSV("dataset.csv");
            clearCanvas(drawingPanel);
        });
        actionsPanel.add(saveBtn);
        
        // Train Button
        JButton trainBtn = createStyledButton("Train MLP", buttonFont);
        trainBtn.setPreferredSize(buttonSize);
        trainBtn.addActionListener(e -> trainModel());
        actionsPanel.add(trainBtn);
        
        // Save Test Button
        JButton saveTestBtn = createStyledButton("Save to Test", buttonFont);
        saveTestBtn.setPreferredSize(buttonSize);
        saveTestBtn.addActionListener(e -> {
            saveToCSV("dataset_test.csv");
            clearCanvas(drawingPanel);
        });
        actionsPanel.add(saveTestBtn);
        
        // Test Button
        JButton testBtn = createStyledButton("Test Model", buttonFont);
        testBtn.setPreferredSize(buttonSize);
        testBtn.addActionListener(e -> testAction());
        actionsPanel.add(testBtn);

        controlPanel.add(actionsPanel);
        
        // Radio buttons for label selection
        JPanel radioPanel = new JPanel();
        radioPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 60, 20));
        radioPanel.setOpaque(false);
        TitledBorder titledBorder = BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(new Color(150, 150, 150), 1),
            "Wybierz litere",
            TitledBorder.CENTER,
            TitledBorder.TOP,
            new Font("Arial", Font.BOLD, 20)
        );
        titledBorder.setTitleColor(Color.BLACK);
        radioPanel.setBorder(titledBorder);

        Font radioFont = new Font("Arial", Font.BOLD, 36);
        
        eRadio = new JRadioButton("e");
        eRadio.setFont(radioFont);
        eRadio.setOpaque(false);
        eRadio.setForeground(Color.BLACK);
        
        lRadio = new JRadioButton("l");
        lRadio.setFont(radioFont);
        lRadio.setOpaque(false);
        lRadio.setForeground(Color.BLACK);
        
        fRadio = new JRadioButton("f");
        fRadio.setFont(radioFont);
        fRadio.setOpaque(false);
        fRadio.setForeground(Color.BLACK);

        ButtonGroup group = new ButtonGroup();
        group.add(eRadio);
        group.add(lRadio);
        group.add(fRadio);

        radioPanel.add(eRadio);
        radioPanel.add(lRadio);
        radioPanel.add(fRadio);

        controlPanel.add(radioPanel);
        
        // Add some vertical glue to push everything up
        controlPanel.add(Box.createVerticalGlue());
        
        return controlPanel;
    }
    
    private JButton createStyledButton(String text, Font font) {
        JButton button = new JButton(text);
        button.setFont(font);
        button.setBackground(Color.BLACK);
        button.setFocusPainted(false);
        return button;
    }

    private void clearCanvas(JPanel drawingPanel) {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    private void recognizeSymbol() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this, 
                "You need to train the model first or load an existing one!", 
                "Not Ready", 
                JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        readPixelsFromCanvas();

        float[][] centeredPixels = centerImage(pixels, GRID);
        if (isEmptyDrawing(centeredPixels)) {
            JOptionPane.showMessageDialog(this, 
                "No drawing detected. Please draw something first.", 
                "Empty Drawing", 
                JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        float[] inputVec = convertToFloatVector(centeredPixels);
        PredictionResult result = mlpModel.predict(inputVec);
        String symbol = indexToSymbol(result.predictedIndex);

        if (result.isUncertain || result.confidence < 0.7f) {
            JOptionPane.showMessageDialog(this, 
                "The model is not confident about this character.", 
                "Uncertain Prediction", 
                JOptionPane.INFORMATION_MESSAGE);
        } else {
            String confidenceText = String.format("%.2f", result.confidence * 100);
            JOptionPane.showMessageDialog(this, 
                "The model predicts: " + symbol + "\nConfidence: " + confidenceText + "%", 
                "Prediction Result", 
                JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void saveToCSV(String csvFile) {
        readPixelsFromCanvas();
        String label = getSelectedLabel();
        if (label == null) {
            JOptionPane.showMessageDialog(this, 
                "Please select a character (e/l/f) before saving.", 
                "Selection Required", 
                JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        CSVUtils.savePixelsToCSV(label, pixels, GRID, csvFile);
        JOptionPane.showMessageDialog(this, 
            "Sample saved successfully to " + csvFile, 
            "Saved", 
            JOptionPane.INFORMATION_MESSAGE);
    }

    private void trainModel() {
        setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        
        SwingWorker<Void, Void> worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground() {
                mlpModel = CSVUtils.trainMLPFromCSV("dataset.csv", GRID);
                if (mlpModel != null) {
                    ModelUtils.saveModel(mlpModel, "mlpModel.bin");
                }
                return null;
            }
            
            @Override
            protected void done() {
                setCursor(Cursor.getDefaultCursor());
                if (mlpModel != null) {
                    JOptionPane.showMessageDialog(AppFrame.this, 
                        "The model has been successfully trained and saved (mlpModel.bin)!", 
                        "Training Complete", 
                        JOptionPane.INFORMATION_MESSAGE);
                } else {
                    JOptionPane.showMessageDialog(AppFrame.this, 
                        "Training failed. Please check the dataset file.", 
                        "Training Error", 
                        JOptionPane.ERROR_MESSAGE);
                }
            }
        };
        
        worker.execute();
    }

    private void testAction() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(this, 
                "You need to train the model first or load an existing one!", 
                "Not Ready", 
                JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        String testCsvFile = "dataset_test.csv";
        File f = new File(testCsvFile);
        if (!f.exists()) {
            JOptionPane.showMessageDialog(this, 
                "Test dataset file not found: " + testCsvFile, 
                "Missing File", 
                JOptionPane.ERROR_MESSAGE);
            return;
        }

        setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        
        SwingWorker<Float, Void> worker = new SwingWorker<>() {
            @Override
            protected Float doInBackground() {
                return CSVUtils.testMLPFromCSV(testCsvFile, mlpModel, GRID);
            }
            
            @Override
            protected void done() {
                setCursor(Cursor.getDefaultCursor());
                try {
                    float accuracy = get();
                    JOptionPane.showMessageDialog(AppFrame.this, 
                        "Classification accuracy: " + Math.round(accuracy * 100) + "%\n" + 
                        "The model correctly identified " + Math.round(accuracy * 100) + " out of 100 test images",
                        "Test Results",
                        JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception e) {
                    e.printStackTrace();
                    JOptionPane.showMessageDialog(AppFrame.this, 
                        "An error occurred during testing.", 
                        "Error", 
                        JOptionPane.ERROR_MESSAGE);
                }
            }
        };
        
        worker.execute();
    }

    private boolean isEmptyDrawing(float[][] pix) {
        int count = 0;
        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                if (pix[y][x] > 0.05f) count++;
            }
        }
        return count < GRID * GRID * 0.01;
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
                double ratio = blackCount / (double)(cellSize * cellSize);
                pixels[y][x] = (float) ratio;
            }
        }
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

        if (!hasContent) return pix;

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

    private float[] convertToFloatVector(float[][] arr) {
        float[] vec = new float[GRID * GRID];
        int k = 0;
        for (int y = 0; y < GRID; y++)
            for (int x = 0; x < GRID; x++)
                vec[k++] = arr[y][x];
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
            case 0: return "e";
            case 1: return "l";
            case 2: return "f";
            default: return "?";
        }
    }
}

