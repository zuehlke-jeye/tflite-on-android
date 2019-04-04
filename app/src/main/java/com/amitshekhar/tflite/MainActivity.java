package com.amitshekhar.tflite;

import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.ShapeDrawable;
import android.os.Bundle;
import android.os.Environment;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import com.wonderkiln.camerakit.CameraKit;
import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.io.File;

public class MainActivity extends AppCompatActivity {
    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final String MODEL_PATH = "facenet.tflite";
    private static final boolean QUANT = true;
    private static final int INPUT_SIZE = 160;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private CameraView cameraView;
    private FloatingActionButton fab;
    private ShapeDrawable boundingBox;
    private View boundingBoxView;

    private FaceClusterer clusterer;

    private void writeImg(Bitmap bitmap) {
        File applicationDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS); //getApplication().getFilesDir();

        File outFile = new File(applicationDir, "bla.jpg");

        try (FileOutputStream out = new FileOutputStream(outFile)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public  boolean requestRuntimePermissions() {
        int result = ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (result == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {

            if (ActivityCompat.shouldShowRequestPermissionRationale(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                Toast.makeText(this, "Write External Storage permission allows us to do store images. Please allow this permission in App Settings.", Toast.LENGTH_LONG).show();
            } else {
                ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
            }
        }
        return true;
    }

    private void handleRecognition(Classifier.Recognition recognition) {
        String name = clusterer.query(recognition);

        if (name != null) {
            nameDialog(name, recognition);
        } else {
            promptDialog(recognition);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = findViewById(R.id.cameraView);
        cameraView.setFacing(CameraKit.Constants.FACING_FRONT);
        fab = findViewById(R.id.fab);
        requestRuntimePermissions();

        clusterer = new FaceClusterer();

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {
                Bitmap bitmap = cameraKitImage.getBitmap();

                StringBuilder sb = new StringBuilder();
                sb.append("Bitmap dimensions: Width = ");
                sb.append(bitmap.getWidth());
                sb.append(", Height = ");
                sb.append(bitmap.getHeight());
                Log.d("fuck", sb.toString());

                bitmap = Bitmap.createBitmap(bitmap, 380, 1200, 1400, 1400);
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
                
                //writeImg(bitmap);

                final Classifier.Recognition result = classifier.recognizeImage(bitmap);


                handleRecognition(result);


                // TODO: do something with result
            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }
        });

        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                cameraView.captureImage();
            }
        });

        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            INPUT_SIZE,
                            QUANT);
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void nameDialog(final String name, final Classifier.Recognition recognition) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Hi " + name + "!");

        builder.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                Log.d("fuck", "Add named cluster");
                clusterer.add(name, recognition);
            }
        });

        builder.show();
    }

    private void promptDialog(final Classifier.Recognition recognition) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);

        builder.setTitle("Add Person");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        builder.setPositiveButton("Add", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                String name = input.getText().toString();
                clusterer.add(name, recognition);
            }
        });

        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                Log.d("fuck", "Cancel adding named cluster");
            }
        });

        builder.show();
    }
}
