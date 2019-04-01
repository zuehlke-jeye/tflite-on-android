package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        private byte data[][];

        public Recognition(final byte data[][]) {
            this.data = data;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();

            try {
                for (int i = 0; i < this.data[0].length; i++) {
                    sb.append(this.data[0][i]);
                }

            } catch (Exception e) {
                sb.append(e.toString());
            }

            return sb.toString();
        }
    }


    Classifier.Recognition recognizeImage(Bitmap bitmap);

    void close();
}
