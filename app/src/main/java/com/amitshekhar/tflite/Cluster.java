package com.amitshekhar.tflite;

import java.util.ArrayList;
import java.util.List;

public class Cluster {
    private static final float NEAR_THRESHOLD = 1.5f;
    private static final float FAR_THRESHOLD = 2.1f;

    private List<Classifier.Recognition> elements;

    public Cluster() {
        elements = new ArrayList<>();
    }

    public boolean isMember(Classifier.Recognition recognition) {
        // heuristic: either there is an element, which is very close
        // or there are at least two, which are reasonably close.

        int farHits = 0;

        for (Classifier.Recognition element : elements) {
            float l2d = element.l2distance(recognition);

            if (l2d <= NEAR_THRESHOLD) {
                return true;
            }

            if (l2d <= FAR_THRESHOLD) {
                farHits++;

                if (farHits >= 2) {
                    return true;
                }
            }
        }

        return false;
    }

    public void add(Classifier.Recognition recognition) {
        elements.add(recognition);
    }
}
