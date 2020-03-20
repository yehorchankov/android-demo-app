//
// Created by yehor on 19.03.20.
//

#include <android/bitmap.h>
#include <jni.h>
#include <string.h>
#include <memory>

#include "ultraface.hpp"


extern "C"
JNIEXPORT jlong JNICALL
Java_org_pytorch_demo_FaceDetectorNative_nativeInitFaceDetector(JNIEnv *env, jclass clazz, jint imageWidth,
                                                                jint imageHeight, jint imageChannels) {
    // One instance of the face detector can be shared through multiple sessions
    FaceDetection* fd = new FaceDetection(imageWidth, imageHeight, imageChannels);
    return (jlong) fd;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_org_pytorch_demo_FaceDetectorNative_nativeReleaseFaceDetector(JNIEnv *env, jclass clazz, jlong netPtr) {
    delete (FaceDetection*) netPtr;
    return 0;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_org_pytorch_demo_FaceDetectorNative_nativeFaceDetect(JNIEnv *env, jclass clazz, jlong faceDetector, jfloatArray scores_, jfloatArray boxes_) {

    // Get outputs
    std::vector<std::vector<float>> outFeatures;

    jfloat* scoresPtr = env->GetFloatArrayElements(scores_, NULL);
    auto scoresLen = env->GetArrayLength(scores_);
    std::vector<float> scores(scoresLen);
    for (int i = 0; i < scoresLen; i++) {
        scores[i] = scoresPtr[i];
    }
    outFeatures.push_back(scores);


    jfloat* boxesPtr = env->GetFloatArrayElements(boxes_, NULL);
    auto boxesLen = env->GetArrayLength(boxes_);
    std::vector<float> boxes(boxesLen);
    for (int i = 0; i < boxesLen; i++) {
        boxes[i] = boxesPtr[i];
    }
    outFeatures.push_back(boxes);


    auto fd = (FaceDetection*) faceDetector;
    // Post processing
    std::vector<FaceInfo> bboxCollection;
    std::vector<FaceInfo> faceList;
    fd->filterBboxes(bboxCollection, outFeatures[0], outFeatures[1]);
    fd->nms(bboxCollection, faceList);

    int length = (int) faceList.size() * 5;
    float *floatResult = new float[length];

    for (int i = 0; i < faceList.size(); i++) {
        floatResult[i*5] = faceList[i].x1;
        floatResult[i*5 + 1] = faceList[i].y1;
        floatResult[i*5 + 2] = faceList[i].x2;
        floatResult[i*5 + 3] = faceList[i].y2;
        floatResult[i*5 + 4] = faceList[i].score;
    }

    auto result_ = env->NewFloatArray(length);

    // Cleanup
    env->SetFloatArrayRegion(result_, 0, length, floatResult);
    env->ReleaseFloatArrayElements(scores_, scoresPtr, 0);
    env->ReleaseFloatArrayElements(boxes_, boxesPtr, 0);
    std::vector<float>().swap(scores);
    std::vector<float>().swap(boxes);
    std::vector<FaceInfo>().swap(bboxCollection);
    std::vector<FaceInfo>().swap(faceList);
    delete [] floatResult;

    return result_;
}
