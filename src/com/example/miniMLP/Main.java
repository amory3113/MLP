package com.example.miniMLP;

import javax.swing.SwingUtilities;
import com.example.miniMLP.ui.AppFrame;


public class Main {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(AppFrame::new);
    }
}
