package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        private float d;

        private float[][] data;

        public Recognition(float[][] data, float d) {
            this.d = d;

            this.data = new float[1][512];
            for (int i = 0; i < 512; i++) {
                this.data[0][i] = data[0][i];
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();


            final boolean isSame = (this.d < 1.2f);

            if (isSame) {
                sb.append("I think you are the same person!\n");
            } else {
                sb.append("I don't think you are the same person!\n");
            }

            sb.append(this.d);/*
            sb.append("\n");
            for (int i = 0; i < 512; i++) {
                sb.append(this.data[0][i]);
                sb.append(" ");
            }*/
            return sb.toString();
        }
    }


    Classifier.Recognition recognizeImage(Bitmap bitmap);

    void close();
}
