package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        private float d;

        public Recognition(float d) {
            this.d = d;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(this.d);
            return sb.toString();
        }
    }


    Classifier.Recognition recognizeImage(Bitmap bitmap);

    void close();
}
