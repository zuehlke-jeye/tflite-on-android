package com.amitshekhar.tflite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FaceClusterer {
    private static final float THRESHOLD = 1.8f;

    private Map<String, List<Classifier.Recognition>> clusters;

    public FaceClusterer() {
        this.clusters = new HashMap<>();
    }

    public String query(Classifier.Recognition recognition) {
        for (Map.Entry<String, List<Classifier.Recognition>> entry : clusters.entrySet()) {
            List<Classifier.Recognition> cluster = entry.getValue();

            for (Classifier.Recognition previous : cluster) {
                if (previous.l2distance(recognition) <= THRESHOLD) {
                    return entry.getKey();
                }
            }
        }
        return null;
    }

    public void add(String name, Classifier.Recognition recognition) {
        List<Classifier.Recognition> found = clusters.get(name);

        if (found != null) {
            found.add(recognition);
        } else {
            List<Classifier.Recognition> newCluster = new ArrayList<>();
            newCluster.add(recognition);
            clusters.put(name, newCluster);
        }
    }
}
