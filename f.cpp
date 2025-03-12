//#include <opencv2/opencv.hpp>
//#include <windows.h> // For mouse control on Windows
//#include "mediapipe/framework/port/parse_text_proto.h"
#include <vector>
#include <deque>
#include <iostream>

// MediaPipe headers - adjust based on your installation
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

using namespace cv;
using namespace std;
using namespace mediapipe;

const int SCREEN_W = GetSystemMetrics(SM_CXSCREEN);
const int SCREEN_H = GetSystemMetrics(SM_CYSCREEN);

class HandTracker {
public:
    HandTracker() {
        string proto_config = R"(
            input_stream: "input_video"
            output_stream: "hand_landmarks"
            node {
                calculator: "HandLandmarkCpu"
                input_stream: "IMAGE:input_video"
                output_stream: "LANDMARKS:hand_landmarks"
            }
        )";
        
        // if (!ParseTextProtoOrDie(proto_config, &config)) {
        if (!ParseTextProto(proto_config, &config)) {
            cerr << "Failed to parse config" << endl;
            exit(1);
        }

        graph.Initialize(config).IgnoreError();
        graph.StartRun({}).IgnoreError();
    }

    vector<NormalizedLandmarkList> process(const Mat& frame) {
        Mat rgb_frame;
        cvtColor(frame, rgb_frame, COLOR_BGR2RGB);
        auto input_frame = absl::make_unique<Frame>(ImageFrame(
            ImageFormat::SRGB, frame.cols, frame.rows, rgb_frame.step, rgb_frame.data));
        
        graph.AddPacketToInputStream("input_video", Adopt(input_frame.release())
            .At(Timestamp::PostStream())).IgnoreError();
        
        Packet packet;
        vector<NormalizedLandmarkList> landmarks;
        if (graph.WaitUntilIdle().ok() && graph.GetOutputStream("hand_landmarks")
            .Next(&packet).ok()) {
            landmarks = packet.Get<vector<NormalizedLandmarkList>>();
        }
        return landmarks;
    }

private:
    CalculatorGraph graph;
};

void moveMouse(int x, int y) {
    SetCursorPos(x, y);
}

int main() {
    cout << "Screen Resolution: " << SCREEN_W << "x" << SCREEN_H << endl;

    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "Camera Resolution: " << width << "x" << height << endl;

    if (!cap.isOpened()) {
        cerr << "Error opening camera" << endl;
        return -1;
    }

    HandTracker tracker;
    deque<Point2f> position_buffer(5);

    while (cap.isOpened()) {
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Failed to read frame" << endl;
            break;
        }
        flip(frame, frame, 1);

        auto landmarks = tracker.process(frame);
        int h = frame.rows, w = frame.cols;

        if (!landmarks.empty()) {
            auto& hand = landmarks[0];
            float norm_x = hand.landmark(8).x(); // INDEX_FINGER_TIP
            float norm_y = hand.landmark(8).y();
            
            float pixel_x = norm_x * w;
            float pixel_y = norm_y * h;
            circle(frame, Point(pixel_x, pixel_y), 5, Scalar(0, 255, 0), -1);

            float screen_x = norm_x * SCREEN_W;
            float screen_y = norm_y * SCREEN_H;

            screen_x = max(0.f, min(screen_x, float(SCREEN_W - 1)));
            screen_y = max(0.f, min(screen_y, float(SCREEN_H - 1)));

            position_buffer.push_back(Point2f(screen_x, screen_y));
            float smoothed_x = 0, smoothed_y = 0;
            for (const auto& p : position_buffer) {
                smoothed_x += p.x;
                smoothed_y += p.y;
            }
            smoothed_x /= position_buffer.size();
            smoothed_y /= position_buffer.size();

            moveMouse(smoothed_x, smoothed_y);

            putText(frame, "Norm: (" + to_string(norm_x) + ", " + to_string(norm_y) + ")",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            putText(frame, "Screen: (" + to_string(int(smoothed_x)) + ", " + 
                    to_string(int(smoothed_y)) + ")", Point(10, 60), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        imshow("Hand Gesture Control", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}