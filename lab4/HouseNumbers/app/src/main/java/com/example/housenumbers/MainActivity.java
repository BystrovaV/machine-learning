package com.example.housenumbers;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.net.Uri;
import android.widget.TextView;

import com.yalantis.ucrop.UCrop;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;


public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Uri imageUri;
    private Interpreter tflite;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
        Button btnCamera = findViewById(R.id.btnCamera);
        Button btnGallery = findViewById(R.id.btnGallery);
        Button btnCrop = findViewById(R.id.btnCrop);
        Button btnPredict = findViewById(R.id.btnPredict);

        btnCamera.setOnClickListener(view -> openCamera());
        btnGallery.setOnClickListener(view -> openGallery());
        btnCrop.setOnClickListener(view -> cropImage());
        btnPredict.setOnClickListener(view -> predictNumber());
        checkPermissions();

        try {
            tflite = new Interpreter(loadModelFile());
            tflite.allocateTensors();
        } catch (IOException e) {
            e.printStackTrace();
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void checkPermissions() {
        String[] permissions = {Manifest.permission.CAMERA, Manifest.permission.READ_MEDIA_IMAGES};
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions, 100);
            }
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, 101);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(intent);
    }

    private void cropImage() {
        if (imageUri == null) return;

        File file = new File(imageUri.getPath());
        if (!file.exists()) {
            Log.e("UCrop", "File does not exist: " + imageUri.getPath());
        } else {
            Log.e("UCrop", "exists");
        }

        Uri destinationUri = Uri.fromFile(new File(getCacheDir(), "cropped.jpg"));
        UCrop.of(imageUri, destinationUri)
                .withAspectRatio(1, 1)
                .withMaxResultSize(64, 64)
                .start(this);
    }

    private void predictNumber() {
        if (imageUri == null) return;

        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            String predicted = predict(bitmap);
            textView.setText(predicted);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == 101 && data != null) { // Камера
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                imageUri = getImageUri(photo);
                imageView.setImageBitmap(photo);
            } else if (requestCode == UCrop.REQUEST_CROP) {
                final Uri resultUri = UCrop.getOutput(data);
                if (resultUri != null) {
                    imageUri = resultUri;
                    imageView.setImageURI(resultUri);
                }
            }
        } else if (resultCode == UCrop.RESULT_ERROR) {
            final Throwable cropError = UCrop.getError(data);
            cropError.printStackTrace();
        }
    }

    private final ActivityResultLauncher<Intent> galleryLauncher =
        registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                imageUri = result.getData().getData();
                imageView.setImageURI(imageUri);
            }
        });

    private Uri getImageUri(Bitmap bitmap) {
        File file = new File(getCacheDir(), "photo.jpg");
        try {
            file.createNewFile();
            FileOutputStream fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
            return Uri.fromFile(file);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 100) {
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    finish(); // Закрываем приложение, если нет разрешений
                }
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        FileInputStream fileInputStream = new FileInputStream(getAssets().openFd("model3.tflite").getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = getAssets().openFd("model3.tflite").getStartOffset();
        long declaredLength = getAssets().openFd("model3.tflite").getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String predict(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, 64, 64, true);

        int imageSize = 64;

        float[][][][] input = new float[1][imageSize][imageSize][1];
        for (int y = 0; y < imageSize; y++) {
            for (int x = 0; x < imageSize; x++) {
                int pixel = bitmap.getPixel(x, y);
                int r = (pixel >> 16 & 0xFF);
                int g = (pixel >> 8 & 0xFF);
                int b = (pixel & 0xFF);
                input[0][y][x][0] = (299 * r + 587 * g + 114 * b) / 1000.0f / 255.0f;
            }
        }

        Map<Integer, Object> outputs = new HashMap<>();
        float[][] output0 = new float[1][11];
        float[][] output1 = new float[1][11];
        float[][] output2 = new float[1][11];
        float[][] output3 = new float[1][11];
        float[][] output4 = new float[1][11];

        outputs.put(0, output0);
        outputs.put(1, output1);
        outputs.put(2, output2);
        outputs.put(3, output3);
        outputs.put(4, output4);

        tflite.runForMultipleInputsOutputs(new Object[]{input}, outputs);
        StringBuilder number = new StringBuilder();
        for (int i = 4; i >= 0; i--) {
            int digit = argmax(((float[][])outputs.get(i))[0]);
            if (digit == 10) {
                number.append(0);
            } else if (digit != 0) {
                number.append(digit);
            }
        }
        return number.toString();
    }
}