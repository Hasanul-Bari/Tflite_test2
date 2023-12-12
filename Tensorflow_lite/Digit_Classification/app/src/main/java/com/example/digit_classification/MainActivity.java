package com.example.digit_classification;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.digit_classification.ml.ModelFashion;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;

    private ImageView imageView;
    private Button uploadButton;
    private Button predictButton;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        uploadButton = findViewById(R.id.uploadButton);
        predictButton = findViewById(R.id.predictButton);
        resultTextView = findViewById(R.id.resultTextView);

        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openGallery();
            }
        });

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                classify();
            }
        });

    }

    private void classify() {
        try {
            ModelFashion model = ModelFashion.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelFashion.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImageUri);
                imageView.setImageBitmap(bitmap);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    private ByteBuffer preprocessImage(Bitmap bitmap) {
        int inputSize = 28; // Assuming your model takes a 28x28 input

        // Resize the bitmap to the model input size (28x28)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);

        // Normalize pixel values to be in the range [0, 1]
        float[] normalizedPixels = normalizePixels(resizedBitmap);

        // Create a ByteBuffer to hold the input data
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(normalizedPixels.length * 4); // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder());

        // Copy the normalized pixel values into the ByteBuffer
        for (float pixelValue : normalizedPixels) {
            inputBuffer.putFloat(pixelValue);
        }

        return inputBuffer;
    }

    private float[] normalizePixels(Bitmap bitmap) {
        int inputSize = 28;
        int[] pixels = new int[inputSize * inputSize];
        float[] normalizedPixels = new float[inputSize * inputSize];

        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);

        // Normalize pixel values to be in the range [0, 1]
        for (int i = 0; i < pixels.length; ++i) {
            final int val = pixels[i];
            normalizedPixels[i] = (Color.red(val) & 0xFF) / 255.0f; // Assuming grayscale image
        }

        return normalizedPixels;
    }

}