package com.example.miniMLP.ui;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.image.BufferedImage;

import com.example.miniMLP.ml.MLP;

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
    private DrawingPanel drawingPanel;
    private AppFunc appFunc;

    public AppFrame() {
        super("MLP");
        setupLookAndFeel();
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(10, 10));

        getContentPane().setBackground(new Color(240, 240, 245));

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

        JPanel rightPanel = createControlPanel();
        add(rightPanel, BorderLayout.CENTER);

        appFunc = new AppFunc(this, drawingPanel, canvas, g2, pixels, eRadio, lRadio, fRadio, GRID);
        setVisible(true);
    }
    
    private void setupLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            System.out.println("Nie można ustawić wyglądu i stylu systemu: " + e);
        }
    }

    private JPanel createControlPanel() {
        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
        controlPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        controlPanel.setOpaque(false);

        Font buttonFont = new Font("Arial", Font.BOLD, 18);
        Dimension buttonSize = new Dimension(250, 80);

        JPanel actionsPanel = new JPanel();
        actionsPanel.setOpaque(false);
        actionsPanel.setLayout(new GridLayout(3, 2, 30, 30));
        actionsPanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 20, 0));

        JButton clearBtn = createStyledButton("Czysc", buttonFont);
        clearBtn.setPreferredSize(buttonSize);
        clearBtn.addActionListener(e -> appFunc.clearCanvas());
        actionsPanel.add(clearBtn);

        JButton recognizeBtn = createStyledButton("Rozpoznać", buttonFont);
        recognizeBtn.setPreferredSize(buttonSize);
        recognizeBtn.addActionListener(e -> appFunc.recognizeSymbol());
        actionsPanel.add(recognizeBtn);

        JButton saveBtn = createStyledButton("Zapisz dane", buttonFont);
        saveBtn.setPreferredSize(buttonSize);
        saveBtn.addActionListener(e -> {
            appFunc.saveToCSV("dataset.csv");
            appFunc.clearCanvas();
        });
        actionsPanel.add(saveBtn);

        JButton trainBtn = createStyledButton("Ucz MLP", buttonFont);
        trainBtn.setPreferredSize(buttonSize);
        trainBtn.addActionListener(e -> appFunc.trainModel());
        actionsPanel.add(trainBtn);

        JButton saveTestBtn = createStyledButton("Zapisz do testu", buttonFont);
        saveTestBtn.setPreferredSize(buttonSize);
        saveTestBtn.addActionListener(e -> {
            appFunc.saveToCSV("dataset_test.csv");
            appFunc.clearCanvas();
        });
        actionsPanel.add(saveTestBtn);

        JButton testBtn = createStyledButton("Test Model", buttonFont);
        testBtn.setPreferredSize(buttonSize);
        testBtn.addActionListener(e -> appFunc.testAction());
        actionsPanel.add(testBtn);

        controlPanel.add(actionsPanel);

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
}
