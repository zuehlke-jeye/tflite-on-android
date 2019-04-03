package com.amitshekhar.tflite;

import android.graphics.Bitmap;

import org.christopherfrantz.dbscan.DBSCANClusteringException;
import org.christopherfrantz.dbscan.DistanceMetric;

import java.util.List;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface Classifier {

    class Recognition {
        private float[][] data;

        public float[][] getData() {
            return this.data;
        }

        public Recognition(float[][] data) {
            this.data = new float[1][512];
            for (int i = 0; i < 512; i++) {
                this.data[0][i] = data[0][i];
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            return sb.toString();
        }

        public float l2distance(Recognition other) {
            float ret = 0.0f;
            for (int i = 0; i < this.data[0].length; i++) {
                float d = this.data[0][i] - other.data[0][i];
                d *= d;

                ret += d;
            }

            return (float) Math.sqrt((double) ret);
        }
    }

    class DistanceMetricRecognition implements DistanceMetric<Recognition> {

        @Override
        public double calculateDistance(Recognition lhs, Recognition rhs) throws DBSCANClusteringException {
            return (double) lhs.l2distance(rhs);
        }
    }

    Classifier.Recognition recognizeImage(Bitmap bitmap);

    void close();
}
