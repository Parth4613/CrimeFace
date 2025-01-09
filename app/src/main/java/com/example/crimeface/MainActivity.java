package com.example.crimeface;

import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.content.pm.PackageManager;
import android.widget.ImageView;
import android.provider.MediaStore;
import org.tensorflow.lite.Interpreter;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_RECOGNIZE_FACE = 2;
    private Button addFaceButton, recognizeButton;
    private Bitmap capturedBitmap;
    private Interpreter tflite;
    private DatabaseHelper dbHelper;
    private ImageView imageViewPerson;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize DatabaseHelper
        dbHelper = new DatabaseHelper(this);

        // Initialize buttons
        addFaceButton = findViewById(R.id.addFaceButton);
        recognizeButton = findViewById(R.id.recognizeButton);
        imageViewPerson = findViewById(R.id.imageViewPerson);


        // Initialize TensorFlow Lite model
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
        }

        // Add Face Button: Open the camera to capture image and then enter details
        addFaceButton.setOnClickListener(v -> captureFaceImage());

        // Recognize Button: Open the camera to capture the image for face recognition
        recognizeButton.setOnClickListener(v -> captureFaceForRecognition());
    }

    // Load the model file from assets
    private ByteBuffer loadModelFile() {
        try (InputStream is = getAssets().open("mobile_face_net.tflite")) {
            byte[] modelBytes = new byte[is.available()];
            is.read(modelBytes);
            ByteBuffer buffer = ByteBuffer.allocateDirect(modelBytes.length);
            buffer.put(modelBytes);
            return buffer;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    // Capture face image using the camera
    private void captureFaceImage() {
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, REQUEST_IMAGE_CAPTURE);
        } else {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    // Capture face image for recognition
    private void captureFaceForRecognition() {
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, REQUEST_RECOGNIZE_FACE);
        } else {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, REQUEST_RECOGNIZE_FACE);
            }
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if ((requestCode == REQUEST_IMAGE_CAPTURE || requestCode == REQUEST_RECOGNIZE_FACE) && resultCode == RESULT_OK) {
            if (data != null && data.getExtras() != null) {
                Bundle extras = data.getExtras();
                capturedBitmap = (Bitmap) extras.get("data");

                if (capturedBitmap != null) {
                    // Update ImageView with the captured image
                    imageViewPerson.setImageBitmap(capturedBitmap);

                    // Resize image for face recognition model
                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(capturedBitmap, 112, 112, true);

                    if (requestCode == REQUEST_IMAGE_CAPTURE) {
                        // Extract feature vector from the captured image using TFLite model
                        float[] featureVector = extractFeatureVector(resizedBitmap);
                        // Prompt user for face details after capturing the image
                        promptForFaceData(featureVector);
                    } else if (requestCode == REQUEST_RECOGNIZE_FACE) {
                        // Recognize the face from the captured image
                        recognizeFace();
                    }
                } else {
                    Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(this, "No data returned from camera", Toast.LENGTH_SHORT).show();
            }
        }
    }


    // Extract feature vector using TensorFlow Lite model
    private float[] extractFeatureVector(Bitmap bitmap) {
        // Prepare input buffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 112 * 112 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        // Convert bitmap to float buffer (normalized RGB)
        int[] intValues = new int[112 * 112];
        bitmap.getPixels(intValues, 0, 112, 0, 0, 112, 112);
        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            inputBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);  // Red
            inputBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);   // Green
            inputBuffer.putFloat((val & 0xFF) / 255.0f);          // Blue
        }

        // Prepare output buffer
        float[][] output = new float[1][192];  // The feature vector size for the model

        // Run inference
        tflite.run(inputBuffer, output);

        return output[0];
    }

    // Prompt the user for face details after capturing the image
    private void promptForFaceData(float[] featureVector) {
        View dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_add_face, null);
        EditText nameInput = dialogView.findViewById(R.id.nameInput);
        EditText firInput = dialogView.findViewById(R.id.firInput);
        EditText caseDescriptionInput = dialogView.findViewById(R.id.caseDescriptionInput);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("Add Face")
                .setView(dialogView)
                .setPositiveButton("Add", (dialogInterface, i) -> {
                    String name = nameInput.getText().toString().trim();
                    String firNo = firInput.getText().toString().trim();
                    String caseDescription = caseDescriptionInput.getText().toString().trim();

                    if (!name.isEmpty() && !firNo.isEmpty() && !caseDescription.isEmpty()) {
                        saveFaceData(name, firNo, caseDescription, featureVector);
                        Toast.makeText(this, "Face added successfully!", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(this, "All fields are required!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel", null)
                .create();

        dialog.show();
    }

    // Save face data to the database
    private void saveFaceData(String name, String firNo, String caseDescription, float[] featureVector) {
        SQLiteDatabase database = dbHelper.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(DatabaseHelper.COLUMN_NAME, name);
        values.put(DatabaseHelper.COLUMN_FIR, firNo);
        values.put(DatabaseHelper.COLUMN_CASE_DESCRIPTION, caseDescription);
        values.put(DatabaseHelper.COLUMN_FEATURE_VECTOR, convertFeatureVectorToByteArray(featureVector));

        long id = database.insert(DatabaseHelper.TABLE_NAME, null, values);
        if (id != -1) {
            Toast.makeText(this, "Face data saved successfully", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Error saving face data", Toast.LENGTH_SHORT).show();
        }
        database.close();
    }

    private byte[] convertFeatureVectorToByteArray(float[] featureVector) {
        ByteBuffer buffer = ByteBuffer.allocate(featureVector.length * 4);
        for (float value : featureVector) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }

    // Recognize face from captured image
    private void recognizeFace() {
        SQLiteDatabase database = dbHelper.getReadableDatabase();

        if (capturedBitmap != null) {
            // Resize and extract feature vector from the captured image
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(capturedBitmap, 112, 112, true);
            float[] capturedVector = extractFeatureVector(resizedBitmap);

            Cursor cursor = database.query(DatabaseHelper.TABLE_NAME, null, null, null, null, null, null);
            boolean matchFound = false;

            // References to TextViews
            TextView nameTextView = findViewById(R.id.matchedName);
            TextView firTextView = findViewById(R.id.matchedFIR);
            TextView caseDescriptionTextView = findViewById(R.id.matchedCaseDescription);

            while (cursor.moveToNext()) {
                // Get stored data from the database
                String name = cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_NAME));
                String firNo = cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_FIR));
                String caseDescription = cursor.getString(cursor.getColumnIndex(DatabaseHelper.COLUMN_CASE_DESCRIPTION));
                byte[] featureVectorBlob = cursor.getBlob(cursor.getColumnIndex(DatabaseHelper.COLUMN_FEATURE_VECTOR));
                float[] storedVector = convertByteArrayToFeatureVector(featureVectorBlob);

                // Calculate cosine similarity between captured and stored feature vectors
                float similarity = calculateCosineSimilarity(capturedVector, storedVector);

                if (similarity > 0.65) {
                    // Update the TextViews with the matched details
                    nameTextView.setText("Name: " + name);
                    firTextView.setText("FIR: " + firNo);
                    caseDescriptionTextView.setText("Case Description: " + caseDescription);

                    matchFound = true;
                    break;
                }
            }

            if (!matchFound) {
                // Clear the TextViews if no match is found
                nameTextView.setText("Name: -");
                firTextView.setText("FIR: -");
                caseDescriptionTextView.setText("Case Description: No match found");
            }

            cursor.close();
            database.close();
        } else {
            Toast.makeText(this, "No image captured!", Toast.LENGTH_SHORT).show();
        }
    }


    // Calculate cosine similarity between two feature vectors
    private float calculateCosineSimilarity(float[] a, float[] b) {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    // Convert byte array to feature vector
    private float[] convertByteArrayToFeatureVector(byte[] blob) {
        FloatBuffer buffer = ByteBuffer.wrap(blob).asFloatBuffer();
        float[] vector = new float[buffer.remaining()];
        buffer.get(vector);
        return vector;
    }
}
