# TensorFlow Lite Products Classification 

### Overview

This is a camera app that continuously classifies the products in the frames seen by your device's back camera, with the option to use a original or a quantized model model trained on MobileNetV2.
These instructions walk you through building and running the app on an Android device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

This application should be run on a physical Android device.
<div align="center">
  <img src="example/main_window.jpg?raw=true" alt="Screenshot with controls" width="300"/>
<p float="left">
  <img src="example/menu_original_model.jpg?raw=true" alt="Screenshot without controls" width="250"/>
  <img src="example/menu_quantized_model.jpg?raw=true" alt="Screenshot without controls" width="250"/>
</p>
</div>
This sample demonstrates how to use TensorFlow Lite with Java.

## Build the demo using Android Studio

### Prerequisites

* The **[Android Studio](https://developer.android.com/studio/index.html)**
    IDE (Android Studio 2021.2.1 or newer). This sample has been tested on
    Android Studio Chipmunk

* A physical Android device with a minimum OS version of SDK 23 (Android 6.0 -
    Marshmallow) with developer mode enabled. The process of enabling developer
    mode may vary by device.

### Building

* Open Android Studio. From the Welcome screen, select Open an existing
    Android Studio project.

* From the Open File or Project window that appears, navigate to and select
    the directory.
    Click OK.

* If it asks you to do a Gradle Sync, click OK.

* With your Android device connected to your computer and developer mode
    enabled, click on the green Run arrow in Android Studio.

### Models used

You can find used models in the `app/src/main/assets` directory. The models are trained on the MobileNetV2 architecture.