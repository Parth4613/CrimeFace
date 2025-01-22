package com.example.crimeface;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.content.pm.PackageManager;
import android.provider.MediaStore;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.tensorflow.lite.Interpreter;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_RECOGNIZE_FACE = 2;
    private Button addFaceButton, recognizeButton;
    private Bitmap capturedBitmap;
    private Interpreter tflite;
    private ImageView imageViewPerson;

    // Firebase reference
    private DatabaseReference firebaseDatabaseReference;

    // To keep track of images
    private int imageCaptureStep = 0; // 0: Front, 1: Left, 2: Right, 3: Eyes Closed
    private float[] featureVectorFront, featureVectorLeft, featureVectorRight, featureVectorEyesClosed;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize Firebase
        firebaseDatabaseReference = FirebaseDatabase.getInstance().getReference("faceData");

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

        // Add Face Button: Open the camera to capture multiple images and then enter details
        addFaceButton.setOnClickListener(v -> startImageCaptureProcess());

        // Recognize Button: Open the camera to capture the image for face recognition
        recognizeButton.setOnClickListener(v -> captureFaceForRecognition());
    }

    // Start the image capture process for 4 types of images
    private void startImageCaptureProcess() {
        imageCaptureStep = 0;
        promptForImageCapture();
    }

    private void promptForImageCapture() {
        String message = switch (imageCaptureStep) {
            case 0 -> "Please capture the front face.";
            case 1 -> "Please capture the left face.";
            case 2 -> "Please capture the right face.";
            case 3 -> "Please capture the front face with eyes closed.";
            default -> null;
        };

        if (message != null) {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
            captureFaceImage();
        } else {
            Toast.makeText(this, "Image capture complete. Please provide details.", Toast.LENGTH_SHORT).show();
            promptForFaceData();
        }
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
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK && data != null && data.getExtras() != null) {
            Bundle extras = data.getExtras();
            capturedBitmap = (Bitmap) extras.get("data");

            if (capturedBitmap != null) {
                imageViewPerson.setImageBitmap(capturedBitmap); // Update the ImageView with the captured image
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(capturedBitmap, 112, 112, true);

                float[] featureVector = extractFeatureVector(resizedBitmap);

                switch (imageCaptureStep) {
                    case 0 -> featureVectorFront = featureVector;
                    case 1 -> featureVectorLeft = featureVector;
                    case 2 -> featureVectorRight = featureVector;
                    case 3 -> featureVectorEyesClosed = featureVector;
                }

                imageCaptureStep++;
                promptForImageCapture();
            } else {
                Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_RECOGNIZE_FACE && resultCode == RESULT_OK && data != null && data.getExtras() != null) {
            Bundle extras = data.getExtras();
            capturedBitmap = (Bitmap) extras.get("data");

            if (capturedBitmap != null) {
                imageViewPerson.setImageBitmap(capturedBitmap); // Ensure the ImageView is updated here too
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(capturedBitmap, 112, 112, true);
                recognizeFace(resizedBitmap);
            } else {
                Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show();
            }
        }
    }


    // Extract feature vector using TensorFlow Lite model
    private float[] extractFeatureVector(Bitmap bitmap) {
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 112 * 112 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[112 * 112];
        bitmap.getPixels(intValues, 0, 112, 0, 0, 112, 112);
        for (int i = 0; i < intValues.length; i++) {
            final int val = intValues[i];
            inputBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);  // Red
            inputBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);   // Green
            inputBuffer.putFloat((val & 0xFF) / 255.0f);          // Blue
        }

        float[][] output = new float[1][192];
        tflite.run(inputBuffer, output);

        return output[0];
    }

    // Recognize face using Firebase
    private void recognizeFace(Bitmap resizedBitmap) {
        float[] capturedVector = extractFeatureVector(resizedBitmap);

        // Firebase recognition
        firebaseDatabaseReference.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                List<FaceData> faceDataList = new ArrayList<>();
                for (DataSnapshot dataSnapshot : snapshot.getChildren()) {
                    FaceData faceData = dataSnapshot.getValue(FaceData.class);
                    if (faceData != null && faceData.FeatureVector != null) {
                        faceDataList.add(faceData);
                    }
                }

                findMatchingFace(capturedVector, faceDataList);
            }

            @Override
            public void onCancelled(DatabaseError error) {
                Toast.makeText(MainActivity.this, "Error fetching data from Firebase: " + error.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Find matching face
    private void findMatchingFace(float[] capturedVector, List<FaceData> faceDataList) {
        TextView nameTextView = findViewById(R.id.matchedName);
        TextView firTextView = findViewById(R.id.matchedFIR);
        TextView caseDescriptionTextView = findViewById(R.id.matchedCaseDescription);

        FaceData bestMatch = null;
        float bestSimilarity = 0f;

        for (FaceData faceData : faceDataList) {
            float[] vectors[] = new float[][]{
                    decodeFeatureVectorFromBase64(faceData.FeatureVector),
                    decodeFeatureVectorFromBase64(faceData.FeatureVectorL),
                    decodeFeatureVectorFromBase64(faceData.FeatureVectorR),
                    decodeFeatureVectorFromBase64(faceData.FeatureVectorEC)
            };
            for (float[] storedVector : vectors) {
                float similarity = calculateCosineSimilarity(capturedVector, storedVector);
                if (similarity > bestSimilarity && similarity > 0.65) {
                    bestSimilarity = similarity;
                    bestMatch = faceData;
                }
            }
        }

        if (bestMatch != null) {
            nameTextView.setText("Name: " + bestMatch.Name);
            firTextView.setText("FIR: " + bestMatch.FIRNo);
            caseDescriptionTextView.setText("Case Description: " + bestMatch.CaseDescription);
        } else {
            nameTextView.setText("Name: -");
            firTextView.setText("FIR: -");
            caseDescriptionTextView.setText("Case Description: No match found");
        }
    }

    // Decode Base64 feature vector
    private float[] decodeFeatureVectorFromBase64(String base64String) {
        if (base64String == null || base64String.isEmpty()) return null;

        byte[] bytes = Base64.decode(base64String, Base64.DEFAULT);
        FloatBuffer buffer = ByteBuffer.wrap(bytes).asFloatBuffer();
        float[] vector = new float[buffer.remaining()];
        buffer.get(vector);
        return vector;
    }

    // Calculate cosine similarity between two feature vectors
    private float calculateCosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null) return 0f;

        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    // Prompt for adding details after capturing all images
    private void promptForFaceData() {
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
                        saveFaceDataToFirebase(name, firNo, caseDescription);
                        Toast.makeText(this, "Face data saved successfully!", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(this, "All fields are required!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel", null)
                .create();

        dialog.show();
    }

    // Save face data to Firebase
    private void saveFaceDataToFirebase(String name, String firNo, String caseDescription) {
        String base64FeatureVectorFront = encodeFeatureVectorToBase64(featureVectorFront);
        String base64FeatureVectorLeft = encodeFeatureVectorToBase64(featureVectorLeft);
        String base64FeatureVectorRight = encodeFeatureVectorToBase64(featureVectorRight);
        String base64FeatureVectorEyesClosed = encodeFeatureVectorToBase64(featureVectorEyesClosed);

        FaceData faceData = new FaceData(name, firNo, caseDescription, base64FeatureVectorFront, base64FeatureVectorLeft, base64FeatureVectorRight, base64FeatureVectorEyesClosed);

        firebaseDatabaseReference.push().setValue(faceData)
                .addOnSuccessListener(aVoid -> Toast.makeText(this, "Face data saved to Firebase", Toast.LENGTH_SHORT).show())
                .addOnFailureListener(e -> Toast.makeText(this, "Error saving face data", Toast.LENGTH_SHORT).show());
    }

    // Encode feature vector to Base64
    private String encodeFeatureVectorToBase64(float[] featureVector) {
        ByteBuffer buffer = ByteBuffer.allocate(featureVector.length * 4);
        for (float value : featureVector) {
            buffer.putFloat(value);
        }
        return Base64.encodeToString(buffer.array(), Base64.DEFAULT);
    }
}
