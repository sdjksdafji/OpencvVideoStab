//#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/video.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

//const int SMOOTHING_RADIUS = 15; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam {
    TransformParam() {}

    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory {
    Trajectory() {}

    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    // "+"
    friend Trajectory operator+(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
    }

    //"-"
    friend Trajectory operator-(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
    }

    //"*"
    friend Trajectory operator*(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
    }

    //"/"
    friend Trajectory operator/(const Trajectory &c1, const Trajectory &c2) {
        return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
    }

    //"="
    Trajectory operator=(const Trajectory &rx) {
        x = rx.x;
        y = rx.y;
        a = rx.a;
        return Trajectory(x, y, a);
    }

    double x;
    double y;
    double a; // angle
};


static void download(const cuda::GpuMat &d_mat, vector<Point2f> &vec) {
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void *) &vec[0]);
    d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

//
int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "./VideoStab [video.avi]" << endl;
        return 0;
    }
    // For further analysis
    ofstream out_transform("prev_to_cur_transformation.txt");
    ofstream out_trajectory("trajectory.txt");
    ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
    ofstream out_new_transform("new_prev_to_cur_transformation.txt");

    VideoCapture cap(argv[1]);
    assert(cap.isOpened());

    Mat curCpu, cur2Cpu;
    cuda::GpuMat cur, cur_grey;
    cuda::GpuMat prev, prev_grey;

    cap >> curCpu;//get the first frame.ch
    prev.upload(curCpu);
    cuda::cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector<TransformParam> prev_to_cur_transform; // previous to current
    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;
    // Step 2 - Accumulate the transformations to get the image trajectory
    vector<Trajectory> trajectory; // trajectory at all frames
    //
    // Step 3 - Smooth out the trajectory using an averaging window
    vector<Trajectory> smoothed_trajectory; // trajectory at all frames
    Trajectory X;//posteriori state estimate
    Trajectory X_;//priori estimate
    Trajectory P;// posteriori estimate error covariance
    Trajectory P_;// priori estimate error covariance
    Trajectory K;//gain
    Trajectory z;//actual measurement
    double pstd = 4e-3;//can be changed
    double cstd = 1;//can be changed, higher the measurement noise, more stable the image is
    double resetRatio = 0.05; // can be changed, if the movement is larger than this ratio in any direction, the kalman
    // filter will be reset. This helps to display image when there are rapid movements
    Trajectory Q(pstd, pstd, pstd);// process noise covariance
    Trajectory R(cstd, cstd, cstd);// measurement noise covariance
    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector<TransformParam> new_prev_to_cur_transform;
    //
    // Step 5 - Apply the new transformation to the video
    //cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    Mat T(2, 3, CV_64F);

    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
    VideoWriter outputVideo;
    outputVideo.open("compare.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 24, Size(cur.rows, cur.cols * 2 + 10),
                     true);
    //
    int k = 1;
    int max_frames = cap.get(CAP_PROP_FRAME_COUNT);
    Mat last_T;

    // prev_grey.type() = 0
    Ptr<cuda::CornersDetector> cornerDetector = cuda::createGoodFeaturesToTrackDetector(0, 20000, 0.01, 30);
    Ptr<cuda::SparsePyrLKOpticalFlow> pfDetector = cuda::SparsePyrLKOpticalFlow::create();

    while (true) {

        cap >> curCpu;
        if (curCpu.data == NULL) {
            break;
        }

        cur.upload(curCpu);
        cuda::cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // vector from prev to cur
        vector<Point2f> prevCornerVector, currentCornerVector;
        vector<Point2f> prevCornerVectorClean, currentCornerVectorClean;
        vector<uchar>  statusVector;
        vector<float> err;

        cuda::GpuMat prevCorner, currentCorner, cornerDiff, cornerDiffMinusMean, status, validStatusMask, closeToMeanMasks, tmpMask, finalMask;

        int type = CV_32FC1;

        cornerDetector->detect(prev_grey, prevCorner);
        pfDetector->calc(prev_grey, cur_grey, prevCorner, currentCorner, status);

        // -------------------------------------------DEBUG-------------------------------------------------------------
//        download(prevCorner, prevCornerVector);
//        download(currentCorner, currentCornerVector);
//        download(status, statusVector);
//
////        cout << "prevCorner.size(w, h): " << prevCorner.size().width << ", " << prevCorner.size().height
////             << " (r,c)" << prevCorner.rows << ", " << prevCorner.cols << " " << prevCorner.type() << endl;
////
////        cout << "status.size(w, h): " << status.size().width << ", " << status.size().height
////             << " (r,c)" << status.rows << ", " << status.cols << " " << status.type() << endl;
//
//        double pre_to_cur_x_avg = 0.0;
//        double pre_to_cur_y_avg = 0.0;
//        double pre_to_cur_x_std = 0.0;
//        double pre_to_cur_y_std = 0.0;
//        // weed out bad matches and calculate mean
//        for (size_t i = 0; i < statusVector.size(); i++) {
//            if (statusVector[i]) {
//                prevCornerVectorClean.push_back(prevCornerVector[i]);
//                currentCornerVectorClean.push_back(currentCornerVector[i]);
////                out_transform << "prevCornerVector:" << prevCornerVector[i].x << "," << prevCornerVector[i].y
////                              << " currentCornerVector" << currentCornerVector[i].x << "," << currentCornerVector[i].y
////                              << endl;
//                // calculate stat
//                pre_to_cur_x_avg += currentCornerVector[i].x - prevCornerVector[i].x;
//                pre_to_cur_y_avg += currentCornerVector[i].y - prevCornerVector[i].y;
//            }
//        }
//        pre_to_cur_x_avg = pre_to_cur_x_avg / currentCornerVectorClean.size();
//        pre_to_cur_y_avg = pre_to_cur_y_avg / currentCornerVectorClean.size();
//
//        // calculate std
//        for(int i =0 ; i<currentCornerVectorClean.size(); i++){
//            double pre_to_cur_x = currentCornerVectorClean[i].x - prevCornerVectorClean[i].x;
//            double pre_to_cur_y = currentCornerVectorClean[i].y - prevCornerVectorClean[i].y;
//            pre_to_cur_x_std += pow(pre_to_cur_x - pre_to_cur_x_avg, 2);
//            pre_to_cur_y_std += pow(pre_to_cur_y - pre_to_cur_y_avg, 2);
//        }
//        pre_to_cur_x_std = sqrt(pre_to_cur_x_std / currentCornerVectorClean.size());
//        pre_to_cur_y_std = sqrt(pre_to_cur_y_std / currentCornerVectorClean.size());

        // -------------------------------------------DEBUG END---------------------------------------------------------

        cuda::compare(status, Scalar(0), validStatusMask, CMP_NE);
        Scalar prevCornerSum = cuda::sum(prevCorner, validStatusMask);
        Scalar currentCornerSum = cuda::sum(currentCorner, validStatusMask);
        Scalar cornerCount = cuda::countNonZero(validStatusMask);
        double prevCornerCentroidX = (double) prevCornerSum[0] / cornerCount[0];
        double prevCornerCentroidY = (double) prevCornerSum[1] / cornerCount[0];
        double currentCornderCentroidX = (double) currentCornerSum[0] / cornerCount[0];
        double currentCornderCentroidY = (double) currentCornerSum[1] / cornerCount[0];
        double diffMeanX = currentCornderCentroidX - prevCornerCentroidX;
        double diffMeanY = currentCornderCentroidY - prevCornerCentroidY;
        cout << "prevCornerSum: " << prevCornerSum << " count: " << cornerCount << " " << prevCornerSum[0] << endl;

        cuda::subtract(currentCorner, prevCorner, cornerDiff);
        cuda::subtract(cornerDiff, Scalar(diffMeanX, diffMeanY), cornerDiffMinusMean);
        Scalar cornerDiffMinusMeanSqrSum = cuda::sqrSum(cornerDiffMinusMean, validStatusMask);
        double diffStdX = sqrt((double) cornerDiffMinusMeanSqrSum[0] / cornerCount[0]);
        double diffStdY = sqrt((double)cornerDiffMinusMeanSqrSum[1] / cornerCount[0]);

        // -------------------------------------------DEBUG-------------------------------------------------------------
//
//        cout << "Mean: " << pre_to_cur_x_avg << "," << pre_to_cur_y_avg << ":" << diffMeanX << diffMeanY << endl;
//        cout << "Std: " << pre_to_cur_x_std << "," << pre_to_cur_y_std << ":" << diffStdX << diffStdY << endl;
//        // find mean of points close to mean
//        int count = 0;
//        double new_mean_dx = 0.0;
//        double new_mean_dy = 0.0;
//        for (int i = 0; i < currentCornerVectorClean.size(); i++) {
//            double pre_to_cur_x = currentCornerVectorClean[i].x - prevCornerVectorClean[i].x;
//            double pre_to_cur_y = currentCornerVectorClean[i].y - prevCornerVectorClean[i].y;
//            if (pre_to_cur_x - pre_to_cur_x_avg < pre_to_cur_x_std
//                && pre_to_cur_y - pre_to_cur_y_avg < pre_to_cur_y_std) {
//                new_mean_dx += pre_to_cur_x;
//                new_mean_dy += pre_to_cur_y;
//                count++;
//            }
//        }
//        new_mean_dx /= count;
//        new_mean_dy /= count;
//
//        // translation + rotation only
//        Mat T = estimateRigidTransform(prevCornerVectorClean, currentCornerVectorClean,
//                                       false); // false = rigid transform, no scaling/shearing
//
//        // in rare cases no transform is found. We'll just use the last known good transform.
//        if (T.data == NULL) {
//            last_T.copyTo(T);
//            k=1;
//        }
//
//        T.copyTo(last_T);
//
//        // original get estimate rigid transformation
////        double dx = T.at<double>(0, 2);
////        double dy = T.at<double>(1, 2);
////        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
//
        // -------------------------------------------DEBUG END---------------------------------------------------------
        vector<cuda::GpuMat> splitCloseToMeanMasks;
        cuda::compare(cornerDiffMinusMean, Scalar(diffStdX, diffStdY), closeToMeanMasks, CMP_LT);

        cuda::split(closeToMeanMasks, splitCloseToMeanMasks);

        cuda::bitwise_and(splitCloseToMeanMasks[0], splitCloseToMeanMasks[1], tmpMask);
        cuda::bitwise_and(tmpMask, validStatusMask, finalMask);
        Scalar closeToCentroidCornerDiffSum = cuda::sum(cornerDiff, finalMask);
        Scalar closeToCentroidCornerCount = cuda::countNonZero(finalMask);
        double dxGpu = closeToCentroidCornerDiffSum[0] / closeToCentroidCornerCount[0];
        double dyGpu = closeToCentroidCornerDiffSum[1] / closeToCentroidCornerCount[0];

        double dx = dxGpu;
        double dy = dyGpu;
        double da = 0;

//        cout << new_mean_dx << "," << new_mean_dy << " : " << dxGpu << "," << dyGpu << endl;

        // -------------------------------------------------------------------------------------------------------------
        // to reset the kalman filter if the camera moves rapidly
        double xResetThreshold = resetRatio * cur.size().width;
        double yResetThreshold = resetRatio * cur.size().height;
        if (dx < -1.0 * xResetThreshold || dx > xResetThreshold
            || dy < -1.0 * yResetThreshold || dy > yResetThreshold) {
            k = 1;
        }
        // -------------------------------------------------------------------------------------------------------------

        out_transform << k << " dx:" << dx << " dy:" << dy << " da:" << da << " points mean x:"
                      << "reduced mean x:" << dxGpu << " y:" << dyGpu << endl;
        //
        // Accumulated frame to frame transform
        x += dx;
        y += dy;
        a += da;
        //trajectory.push_back(Trajectory(x,y,a));
        //
        out_trajectory << k << " " << x << " " << y << " " << a << endl;
        //
        z = Trajectory(x, y, a);
        //
        if (k == 1) {
            // intial guesses
            X = Trajectory(0, 0, 0); //Initial estimate,  set 0
            P = Trajectory(1, 1, 1); //set error variance,set 1
            K = Trajectory(0, 0, 0); // Kalman gain, set just for printing purpose only
            x = 0;
            y = 0;
            a = 0;
        } else {
            //time update（prediction）
            X_ = X; //X_(k) = X(k-1);
            P_ = P + Q; //P_(k) = P(k-1)+Q;
            // measurement update（correction）
            K = P_ / (P_ + R); //gain;K(k) = P_(k)/( P_(k)+R );
            X = X_ + K * (z - X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k));
            P = (Trajectory(1, 1, 1) - K) * P_; //P(k) = (1-K(k))*P_(k);
        }
        //smoothed_trajectory.push_back(X);
        out_smoothed_trajectory << k << " K:" << K.x << "," << K.y << "," << K.a << " x:" << X.x << " y:" << X.y
                                << " a:" << X.a << endl;
        //-
        // target - current
        double diff_x = X.x - x;//
        double diff_y = X.y - y;
        double diff_a = X.a - a;

        dx = diff_x;
        dy = diff_y;
        da = diff_a;

        //new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
        out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
        // Matrix for new prev to cur transformation
//        da = 3.14159265 * 45 / 180; // test: rotate clockwise by 45
        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);

        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;

        cuda::GpuMat cur2;

        cuda::rotate(cur, cur2, cur.size(), 0, dx, dy);

        cur2.download(cur2Cpu);
        // Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(curCpu.rows, curCpu.cols * 2 + 10, cur.type());

        curCpu.copyTo(canvas(Range::all(), Range(0, curCpu.cols)));
        cur2Cpu.copyTo(canvas(Range::all(), Range(cur2Cpu.cols + 10, cur2.cols * 2 + 10)));

        // If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        if (canvas.cols > 1080) {
            resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
        }
        //outputVideo<<canvas;
        imshow("before and after", canvas);

        waitKey(10);
        //
        prev = cur.clone();//cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);

        cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prevCornerVectorClean.size() << endl;
        k++;

    }
    return 0;
}