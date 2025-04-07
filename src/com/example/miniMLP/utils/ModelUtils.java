package com.example.miniMLP.utils;

import com.example.miniMLP.ml.MLP;
import java.io.*;

public class ModelUtils {
    public static void saveModel(MLP mlp, String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(mlp);
            System.out.println("Модель успішно збережена!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MLP loadModel(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            MLP mlp = (MLP) ois.readObject();
            System.out.println("Модель успішно завантажена!");
            return mlp;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
}

