package com.example.miniMLP.ui;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import com.example.miniMLP.ml.MLP;
import com.example.miniMLP.ml.PredictionResult;
import com.example.miniMLP.utils.CSVUtils;
import com.example.miniMLP.utils.ModelUtils;

public class AppFunc {
    private final int GRID;
    private JFrame parentFrame;
    private DrawingPanel drawingPanel;
    private BufferedImage canvas;
    private Graphics2D g2;
    private float[][] pixels;
    private JRadioButton eRadio;
    private JRadioButton lRadio;
    private JRadioButton fRadio;
    private MLP mlpModel;

    public AppFunc(JFrame parentFrame, DrawingPanel drawingPanel, BufferedImage canvas, Graphics2D g2, float[][] pixels, JRadioButton eRadio, JRadioButton lRadio, JRadioButton fRadio, int grid) {
        this.parentFrame = parentFrame;
        this.drawingPanel = drawingPanel;
        this.canvas = canvas;
        this.g2 = g2;
        this.pixels = pixels;
        this.eRadio = eRadio;
        this.lRadio = lRadio;
        this.fRadio = fRadio;
        this.GRID = grid;

        String modelPath = "mlpModel.bin";
        if (new File(modelPath).exists()) {
            mlpModel = ModelUtils.loadModel(modelPath);
        }
    }

    public void clearCanvas() {
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        g2.setColor(Color.BLACK);
        drawingPanel.repaint();
    }

    public void recognizeSymbol() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(parentFrame, "Najpierw musisz wytrenować model!", "Nie gotowy", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        readPixelsFromCanvas();

        float[][] centeredPixels = centerImage(pixels, GRID);
        if (isEmptyDrawing(centeredPixels)) {
            JOptionPane.showMessageDialog(parentFrame, "Modelka nie jest pewna tej postaci.", "Niepewna prognoza", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        float[] inputVec = convertToFloatVector(centeredPixels);
        PredictionResult result = mlpModel.predict(inputVec);
        String symbol = indexToSymbol(result.predictedIndex);

        float entropy = 0;
        if (result.probabilities != null) {
            for (float p : result.probabilities) {
                if (p > 0) entropy -= p * Math.log(p) / Math.log(result.probabilities.length);
            }
        }
        boolean seemsRandom = entropy > 0.7f;
        
        if (result.isUncertain || result.confidence < 0.9f || seemsRandom) {
            JOptionPane.showMessageDialog(parentFrame, "Modelka nie jest pewna tej postaci.", "Niepewna prognoza", JOptionPane.INFORMATION_MESSAGE);
        } else {
            JOptionPane.showMessageDialog(parentFrame, "Model przewiduje: " + symbol, "Wynik przewidywania", JOptionPane.INFORMATION_MESSAGE);
        }
    }

    public void saveToCSV(String csvFile) {
        readPixelsFromCanvas();
        String label = getSelectedLabel();
        if (label == null) {
            JOptionPane.showMessageDialog(parentFrame, "Przed zapisaniem wybierz znak (e/l/f).", "Wybór wymagany", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        CSVUtils.savePixelsToCSV(label, pixels, GRID, csvFile);
    }

    public void trainModel() {
        parentFrame.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        
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
                parentFrame.setCursor(Cursor.getDefaultCursor());
                if (mlpModel != null) {
                    JOptionPane.showMessageDialog(parentFrame, "Model został pomyślnie wytrenowany i zapisany!", "Szkolenie ukończone", JOptionPane.INFORMATION_MESSAGE);
                } else {
                    JOptionPane.showMessageDialog(parentFrame, "Trening nie powiódł się. Sprawdź plik zestawu danych.", "Błąd szkolenia", JOptionPane.ERROR_MESSAGE);
                }
            }
        };
        
        worker.execute();
    }

    public void testAction() {
        if (mlpModel == null) {
            JOptionPane.showMessageDialog(parentFrame, "Najpierw musisz wytrenować model lub załadować istniejący!", "Nie gotowy", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        String testCsvFile = "dataset_test.csv";
        File f = new File(testCsvFile);
        if (!f.exists()) {
            JOptionPane.showMessageDialog(parentFrame, "Nie znaleziono pliku zestawu danych testowych: " + testCsvFile, "Brakujący plik", JOptionPane.ERROR_MESSAGE);
            return;
        }

        parentFrame.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        
        SwingWorker<Float, Void> worker = new SwingWorker<>() {
            @Override
            protected Float doInBackground() {
                return CSVUtils.testMLPFromCSV(testCsvFile, mlpModel, GRID);
            }
            
            @Override
            protected void done() {
                parentFrame.setCursor(Cursor.getDefaultCursor());
                try {
                    float accuracy = get();
                    JOptionPane.showMessageDialog(parentFrame, "Skuteczność sieci: " + Math.round(accuracy * 100) + " z 100 obrazów", "Wyniki testów", JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception e) {
                    e.printStackTrace();
                    JOptionPane.showMessageDialog(parentFrame, "Wystąpił błąd podczas testowania.", "Błąd", JOptionPane.ERROR_MESSAGE);
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
                pixels[y][x] = ratio > 0.2f ? 1.0f : 0.0f;
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
