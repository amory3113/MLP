package com.example.miniMLP.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;

public class DrawingPanel extends JPanel {
    private BufferedImage canvas;
    private Graphics2D g2;
    private Point lastPoint;
    private int drawSize = 13;

    public DrawingPanel(BufferedImage canvas) {
        this.canvas = canvas;
        setPreferredSize(new Dimension(canvas.getWidth(), canvas.getHeight()));
        setBackground(Color.WHITE);
        g2 = canvas.createGraphics();
        g2.setColor(Color.BLACK);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        MouseAdapter mouseAdapter = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastPoint = e.getPoint();
                drawPoint(e.getPoint());
            }
            @Override
            public void mouseDragged(MouseEvent e) {
                Point currentPoint = e.getPoint();
                drawLine(lastPoint, currentPoint);
                lastPoint = currentPoint;
            }
        };
        addMouseListener(mouseAdapter);
        addMouseMotionListener(mouseAdapter);
    }
    
    private void drawPoint(Point p) {
        g2.fillOval(p.x - drawSize/2, p.y - drawSize/2, drawSize, drawSize);
        repaint();
    }
    
    private void drawLine(Point from, Point to) {
        g2.setStroke(new BasicStroke(drawSize, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g2.drawLine(from.x, from.y, to.x, to.y);
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(new Color(240, 240, 240));
        int gridSize = 20;
        for (int x = 0; x < getWidth(); x += gridSize) {
            g.drawLine(x, 0, x, getHeight());
        }
        for (int y = 0; y < getHeight(); y += gridSize) {
            g.drawLine(0, y, getWidth(), y);
        }
        g.drawImage(canvas, 0, 0, null);
    }
}
