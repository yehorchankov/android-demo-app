package org.pytorch.demo.vision;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.FaceDetectorNative;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;
import org.pytorch.demo.TensorImageUtilsCopy;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
    public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

    private static final float[] ULTRANET_NORM_MEAN_RGB = new float[]{127f, 127f, 127f};
    private static final float[] ULTRANET_NORM_STD_RGB = new float[]{1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
    private static final int TOP_K = 3;
    private static final int MOVING_AVG_PERIOD = 10;
    private static final String FORMAT_MS = "%dms";
    private static final String FORMAT_AVG_MS = "avg:%.0fms";

    private static final String FORMAT_FPS = "%.1fFPS";
    public static final String SCORES_FORMAT = "%.2f";

    static class AnalysisResult {

        private final String[] topNClassNames;
        private final float[] topNScores;
        private final long analysisDuration;
        private final long moduleForwardDuration;

        public AnalysisResult(String[] topNClassNames, float[] topNScores,
                              long moduleForwardDuration, long analysisDuration) {
            this.topNClassNames = topNClassNames;
            this.topNScores = topNScores;
            this.moduleForwardDuration = moduleForwardDuration;
            this.analysisDuration = analysisDuration;
        }
    }

    private boolean mAnalyzeImageErrorState;
    private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
    private TextView mFpsText;
    private TextView mMsText;
    private TextView mMsAvgText;
    private Module mModule;
    private Long FaceDetector;
    private String mModuleAssetName;
    private FloatBuffer mInputTensorBuffer;
    private Tensor mInputTensor;
    private long mMovingAvgSum = 0;
    private Queue<Long> mMovingAvgQueue = new LinkedList<>();

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_image_classification;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
                .inflate()
                .findViewById(R.id.image_classification_texture_view);
    }

    @Override
    protected SurfaceView getFaceBoxesSurfaceView() {
        return ((ViewStub) findViewById(R.id.face_boxes_surface_view_stub))
                .inflate()
                .findViewById(R.id.face_boxes_surface_view);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (this.getInfoViewCode() == InfoViewFactory.INFO_VIEW_TYPE_FACE_DETECTION_ULTRANET) {
            this.inputWidth = 320;
            this.inputHeight = 240;
        } else {
            this.inputWidth = 224;
            this.inputHeight = 224;
        }
        super.onCreate(savedInstanceState);

        final ResultRowView headerResultRowView =
                findViewById(R.id.image_classification_result_header_row);
        headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
        headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

        mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
        mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
        mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

        mFpsText = findViewById(R.id.image_classification_fps_text);
        mMsText = findViewById(R.id.image_classification_ms_text);
        mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mMovingAvgSum += result.moduleForwardDuration;
        mMovingAvgQueue.add(result.moduleForwardDuration);
        if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
            mMovingAvgSum -= mMovingAvgQueue.remove();
        }

        for (int i = 0; i < TOP_K; i++) {
            final ResultRowView rowView = mResultRowViews[i];
            rowView.nameTextView.setText(result.topNClassNames[i]);
            rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
                    result.topNScores[i]));
            rowView.setProgressState(false);
        }

        mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
        if (mMsText.getVisibility() != View.VISIBLE) {
            mMsText.setVisibility(View.VISIBLE);
        }
        mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
        if (mFpsText.getVisibility() != View.VISIBLE) {
            mFpsText.setVisibility(View.VISIBLE);
        }

        if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
            float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
            mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
            if (mMsAvgText.getVisibility() != View.VISIBLE) {
                mMsAvgText.setVisibility(View.VISIBLE);
            }
        }
    }

    protected String getModuleAssetName() {
        if (!TextUtils.isEmpty(mModuleAssetName)) {
            return mModuleAssetName;
        }
        final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
                ? moduleAssetNameFromIntent
                : "resnet18.pt";

        return mModuleAssetName;
    }

    @Override
    protected String getInfoViewAdditionalText() {
        return getModuleAssetName();
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mAnalyzeImageErrorState) {
            return null;
        }

        try {
            if (this.getInfoViewCode() == InfoViewFactory.INFO_VIEW_TYPE_FACE_DETECTION_ULTRANET) {
                if (mModule == null) {
                    // Create model and input tensor
                    final String moduleFileAbsoluteFilePath = new File(
                            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
                    mModule = Module.load(moduleFileAbsoluteFilePath);

                    mInputTensorBuffer =
                            Tensor.allocateFloatBuffer(3 * this.inputWidth * this.inputHeight);
                    mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, this.inputHeight, this.inputWidth});
                }

                if (FaceDetector == null) {
                    FaceDetector = FaceDetectorNative.nativeInitFaceDetector(this.inputWidth, this.inputHeight, 3);
                }

                // Handle face detection
                final long startTime = SystemClock.elapsedRealtime();

                TensorImageUtilsCopy.imageYUV420CenterCropToFloatBuffer(
                        image.getImage(), rotationDegrees,
                        this.inputWidth, this.inputHeight,
                        ULTRANET_NORM_MEAN_RGB, ULTRANET_NORM_STD_RGB,
                        mInputTensorBuffer, 0);

                final long moduleForwardStartTime = SystemClock.elapsedRealtime();
                final IValue[] outputTensors = mModule.forward(IValue.from(mInputTensor)).toTuple();
                final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

                final float[] scores = outputTensors[0].toTensor().getDataAsFloatArray();
                final float[] boxes = outputTensors[1].toTensor().getDataAsFloatArray();

                float[] result = FaceDetectorNative.nativeFaceDetect(FaceDetector, scores, boxes);

                final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

                Canvas canvas = faceBoxesViewHolder.lockCanvas();

                canvas.drawColor( 0, PorterDuff.Mode.CLEAR );

                Paint rectPaint = new Paint();
                int canvasWidth = canvas.getWidth();
                int canvasHeight = canvas.getHeight();
                rectPaint.setColor(Color.MAGENTA);
                rectPaint.setStyle(Paint.Style.STROKE);
                rectPaint.setStrokeWidth(5);

                Paint textPaint = new Paint();
                textPaint.setColor(Color.MAGENTA);
                textPaint.setTextSize(37);

                for (int i = 0; i < result.length / 5; i++) {
                    canvas.drawRect(result[i * 5] / inputWidth * canvasWidth,
                            result[i * 5 + 1] / inputHeight * canvasHeight,
                            result[i * 5 + 2] / inputWidth * canvasWidth,
                            result[i * 5 + 3] / inputHeight * canvasHeight, rectPaint);
                    canvas.drawText(String.format("%.3f", result[i * 5 + 4]),
                            result[i * 5] / inputWidth * canvasWidth,
                            result[i * 5 + 1] / inputHeight * canvasHeight, textPaint);
                }

                faceBoxesViewHolder.unlockCanvasAndPost(canvas);

                return new AnalysisResult(new String[] {"Faces 1", "Faces 2", "Faces 3"}, new float[] {0.7f, 0.2f, 0.1f}, moduleForwardDuration, analysisDuration);
            } else {
                if (mModule == null) {
                    // Create model and input tensor
                    final String moduleFileAbsoluteFilePath = new File(
                            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
                    mModule = Module.load(moduleFileAbsoluteFilePath);

                    mInputTensorBuffer =
                            Tensor.allocateFloatBuffer(3 * this.inputWidth * this.inputHeight);
                    mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, this.inputHeight, this.inputWidth});
                }

                // Handle classification
                final long startTime = SystemClock.elapsedRealtime();
                TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                        image.getImage(), rotationDegrees,
                        this.inputWidth, this.inputHeight,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                        mInputTensorBuffer, 0);

                final long moduleForwardStartTime = SystemClock.elapsedRealtime();
                final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
                final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

                final float[] scores = outputTensor.getDataAsFloatArray();
                final int[] ixs = Utils.topK(scores, TOP_K);

                final String[] topKClassNames = new String[TOP_K];
                final float[] topKScores = new float[TOP_K];
                for (int i = 0; i < TOP_K; i++) {
                    final int ix = ixs[i];
                    topKClassNames[i] = Constants.IMAGENET_CLASSES[ix];
                    topKScores[i] = scores[ix];
                }
                final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

                return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration);
            }
        } catch (Exception e) {
            Log.e(Constants.TAG, "Error during image analysis", e);
            mAnalyzeImageErrorState = true;
            runOnUiThread(() -> {
                if (!isFinishing()) {
                    showErrorDialog(v -> ImageClassificationActivity.this.finish());
                }
            });
            return null;
        }
    }

    @Override
    protected int getInfoViewCode() {
        return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mModule != null) {
            mModule.destroy();
        }
        if (FaceDetector != null) {
            FaceDetectorNative.nativeReleaseFaceDetector(FaceDetector);
            FaceDetector = null;
        }
    }
}
