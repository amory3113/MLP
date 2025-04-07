package com.example.miniMLP.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class DrawingPanel extends JPanel {
    private BufferedImage canvas;
    private Graphics2D g2;

    public DrawingPanel(BufferedImage canvas) {
        this.canvas = canvas;
        setPreferredSize(new Dimension(canvas.getWidth(), canvas.getHeight()));

        // Создаём постоянный Graphics2D для холста
        g2 = canvas.createGraphics();
        g2.setColor(Color.BLACK);

        addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                int size = 12;
                // Рисуем чёрной кистью
                g2.fillOval(e.getX() - size / 2, e.getY() - size / 2, size, size);
                repaint();
            }
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        // Отображаем текущее содержимое canvas
        g.drawImage(canvas, 0, 0, null);
    }
}


