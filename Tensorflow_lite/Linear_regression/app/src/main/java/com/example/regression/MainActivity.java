package com.example.regression;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.regression.ml.RegressionModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private TextView outputTextview;
    private Button submitButton;
    private EditText inputEditText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        outputTextview = findViewById(R.id.textview);
        inputEditText = findViewById(R.id.editText);
        submitButton = findViewById(R.id.buttonSubmit);

        submitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Float data = Float.parseFloat(inputEditText.getText().toString());
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1*4);
                byteBuffer.order(ByteOrder.nativeOrder());
                byteBuffer.putFloat(data);
                byteBuffer.rewind();

                try {
                    RegressionModel model = RegressionModel.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 1}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    RegressionModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    //output result in textView
                    float[] data1=outputFeature0.getFloatArray();
                    outputTextview.setText("Predicted value: "+String.valueOf(data1[0]));

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }




            }
        });

    }
}