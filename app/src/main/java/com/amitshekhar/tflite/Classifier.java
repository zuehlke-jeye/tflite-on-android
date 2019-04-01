package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        private byte data[][];
        private double d;

        public Recognition(final byte data[][], double d) {
            this.data = data;
            this.d = d;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();

            sb.append(this.d);

            /*try {
                for (int i = 0; i < this.data[0].length; i++) {
                    sb.append(this.data[0][i]);
                }

            } catch (Exception e) {
                sb.append(e.toString());
            }*/

            return sb.toString();
        }
    }


    Classifier.Recognition recognizeImage(Bitmap bitmap);

    void close();
}
