package org.pytorch.demo;


public class FaceDetectorNative {
    // load libraries
    static {
        System.loadLibrary("corelib");
    }

        //Face detector helpers
    public static native long nativeInitFaceDetector(int imageWidth , int imageHeight, int imageChannels);

    public static native long nativeReleaseFaceDetector(long netPtr);


    //Do face detect
    public static native float[] nativeFaceDetect(long faceDetector, float[] scores, float[] boxes);

}
