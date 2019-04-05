package com.amitshekhar.tflite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FaceClusterer {
    private Map<String, Cluster> clusters;

    public FaceClusterer() {
        this.clusters = new HashMap<>();
    }

    public String query(Classifier.Recognition recognition) {
        for (Map.Entry<String, Cluster> entry : clusters.entrySet()) {
            Cluster cluster = entry.getValue();

            if (cluster.isMember(recognition)) {
                return entry.getKey();
            }
        }
        return null;
    }

    public void add(String name, Classifier.Recognition recognition) {
        Cluster found = clusters.get(name);

        if (found != null) {
            found.add(recognition);
        } else {
            Cluster newCluster = new Cluster();
            newCluster.add(recognition);
            clusters.put(name, newCluster);
        }
    }
}
