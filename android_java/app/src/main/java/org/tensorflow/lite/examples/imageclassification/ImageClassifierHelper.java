/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.imageclassification.fragments.Classifications;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

/** Helper class for wrapping Image Classification actions */
public class ImageClassifierHelper {
    private static final int MODEL_PRODUCTS_QUANTIZED = 1;
    private static final int MODEL_PRODUCTS_113 = 2;
    private static final int MODEL_PRODUCTS_113_QUANTIZED = 3;

    private float threshold;
    private int numThreads;
    private int maxResults;
    private int currentModel;
    private final Context context;
    private final ClassifierListener imageClassifierListener;
    private ImageClassifier imageClassifier;

    private Interpreter tflite;

    /** Helper class for wrapping Image Classification actions */
    public ImageClassifierHelper(Float threshold,
                                 int numThreads,
                                 int maxResults,
                                 int currentModel,
                                 Context context,
                                 ClassifierListener imageClassifierListener) {
        this.threshold = threshold;
        this.numThreads = numThreads;
        this.maxResults = maxResults;
        this.currentModel = currentModel;
        this.context = context;
        this.imageClassifierListener = imageClassifierListener;
        setupImageClassifier();
    }

    public static ImageClassifierHelper create(
            Context context,
            ClassifierListener listener
    ) {
        // load interpreter
        return new ImageClassifierHelper(
                0.5f,
                2,
                3,
                0,
                context,
                listener
        );
    }

    public int getMaxResults() {
        return maxResults;
    }

    public void setMaxResults(int maxResults) {
        this.maxResults = maxResults;
    }

    public void setCurrentModel(int currentModel) {
        this.currentModel = currentModel;
    }

    private void setupImageClassifier() {
        String modelName;
        switch (currentModel) {
            case MODEL_PRODUCTS_QUANTIZED:
                modelName = "quantized_model.tflite";
                break;
            case MODEL_PRODUCTS_113:
                modelName = "original_model_113.tflite";
                break;
            case MODEL_PRODUCTS_113_QUANTIZED:
                modelName = "quantized_model_113.tflite";
                break;
            default:
                modelName = "original_model.tflite";
        }

        Interpreter.Options options = new Interpreter.Options();
        this.tflite = new Interpreter(loadModelFile(modelName), options);
    }

    private MappedByteBuffer loadModelFile(String modelFilename) {
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFilename);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (Exception e) {
            return null;
        }
    }

    public TensorImage loadImage(final Bitmap bitmap) {
        // Load the bitmap into a TensorImage
        TensorImage inputImageBuffer = new TensorImage(DataType.FLOAT32);
        inputImageBuffer.load(bitmap);

        // Define the image processor with normalization parameters
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(127.5f, 127.5f))  // Adjust these values if needed
                        .build();

        // Process the image
        return imageProcessor.process(inputImageBuffer);
    }

    public void classify(Bitmap bitmap) {
        if (imageClassifier == null) {
            setupImageClassifier();
        }

        // Inference time is the difference between the system time at the start and finish of the process
        long inferenceTime = SystemClock.uptimeMillis();

        TensorImage tensorImage = loadImage(bitmap);

        // Prepare the input and output buffers
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32); // Adjust the output size as needed

        // Run the model
        tflite.run(tensorImage.getBuffer(), outputBuffer.getBuffer());

        // Convert result_array to List<Classifications>
        List<Classifications> classificationsList = new ArrayList<>();
        classificationsList.add(new Classifications(convertToCategories(outputBuffer.getFloatArray())));

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime;
        imageClassifierListener.onResults(classificationsList, inferenceTime);
    }

    // Helper method to convert float array to list of Category objects
    private List<Category> convertToCategories(float[] resultArray) {
        List<Category> categories = new ArrayList<>();
        String[] labels = {
                "Bean", "Bitter_Gourd", "Bottle_Gourd",
                "Brinjal", "Broccoli", "Cabbage",
                "Capsicum", "Carrot", "Cauliflower",
                "Cucumber", "Papaya", "Potato",
                "Pumpkin", "Radish", "Tomato"
        };
        for (int i = 0; i < labels.length; i++) {
            Category category = new Category(labels[i], resultArray[i]);
            categories.add(category);
        }
        Collections.sort(categories, (c1, c2) -> Float.compare(c2.getScore(), c1.getScore()));
        return categories;
    }

    public void clearImageClassifier() {
        imageClassifier = null;
    }

    /** Listener for passing results back to calling class */
    public interface ClassifierListener {
        void onError(String error);

        void onResults(List<Classifications> results, long inferenceTime);
    }
}

