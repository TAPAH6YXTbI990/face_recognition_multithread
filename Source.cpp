#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

using namespace cv;
using namespace std;

struct FrameData {
    Mat frame;
    Mat gray;
    vector<Rect> faces;
    vector<vector<Rect>> eyes;
    vector<vector<Rect>> smiles;
    int frameNumber;
};

queue<FrameData> frameQueue;
mutex queueMutex;
condition_variable queueCondVar;
atomic<bool> processingDone(false);
atomic<bool> captureDone(false);
const int MAX_QUEUE_SIZE = 30; 

void processFrames(CascadeClassifier& face_cascade, CascadeClassifier& eye_cascade, CascadeClassifier& smile_cascade) {
    while (true) {
        FrameData data;

        {
            unique_lock<mutex> lock(queueMutex);
            queueCondVar.wait(lock, [] {
                return !frameQueue.empty() || (processingDone && captureDone);
                });
            if (frameQueue.empty() && captureDone) break;
            data = move(frameQueue.front());
            frameQueue.pop();
        }

        face_cascade.detectMultiScale(data.gray, data.faces, 1.1, 4, 0, Size(150, 150));

        data.eyes.resize(data.faces.size());
        data.smiles.resize(data.faces.size());

        for (size_t i = 0; i < data.faces.size(); i++) {
            Mat face_area = data.gray(data.faces[i]);

            eye_cascade.detectMultiScale(face_area, data.eyes[i], 1.27, 5, 0, Size(60, 60));

            smile_cascade.detectMultiScale(face_area, data.smiles[i], 1.27, 20, 0, Size(35, 35));
        }

        for (size_t i = 0; i < data.faces.size(); i++) {
            rectangle(data.frame, data.faces[i], Scalar(255, 0, 0), 2);

            for (size_t j = 0; j < data.eyes[i].size(); j++) {
                rectangle(data.frame,
                    Point(data.faces[i].x + data.eyes[i][j].x, data.faces[i].y + data.eyes[i][j].y),
                    Point(data.faces[i].x + data.eyes[i][j].x + data.eyes[i][j].width,
                        data.faces[i].y + data.eyes[i][j].y + data.eyes[i][j].height),
                    Scalar(0, 255, 0), 2);
            }

            for (size_t j = 0; j < data.smiles[i].size(); j++) {
                rectangle(data.frame,
                    Point(data.faces[i].x + data.smiles[i][j].x, data.faces[i].y + data.smiles[i][j].y),
                    Point(data.faces[i].x + data.smiles[i][j].x + data.smiles[i][j].width,
                        data.faces[i].y + data.smiles[i][j].y + data.smiles[i][j].height),
                    Scalar(0, 0, 255), 2);
            }
        }

        imshow("Recognized faces", data.frame);
        if (waitKey(30) == 'q') {
            processingDone = true;
            break;
        }
    }
}

int main() {
    VideoCapture cap("C:\\Users\\User\\.vscode\\projects\\ZUA.mp4");
    if (!cap.isOpened()) {
        cerr << "Error when uploading the video!" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    int delay = 1000 / fps;

    CascadeClassifier face_cascade, eye_cascade, smile_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Error: couldn't load the face classifier!" << endl;
        return -1;
    }
    if (!eye_cascade.load("haarcascade_eye.xml")) {
        cerr << "Error: couldn't load the eye classifier!" << endl;
        return -1;
    }
    if (!smile_cascade.load("haarcascade_smile.xml")) {
        cerr << "Error: couldn't load the smile classifier!" << endl;
        return -1;
    }

    const int num_processing_threads = 2;

    vector<thread> processingThreads;
    for (int i = 0; i < num_processing_threads; ++i) {
        processingThreads.emplace_back(processFrames, ref(face_cascade), ref(eye_cascade), ref(smile_cascade));
    }

    int frameNumber = 0;
    Mat frame;
    while (!processingDone) {
        bool success = cap.read(frame);
        if (!success) {
            captureDone = true;
            break;
        }

        FrameData data;
        data.frame = frame.clone();
        cvtColor(frame, data.gray, COLOR_BGR2GRAY);
        equalizeHist(data.gray, data.gray);
        data.frameNumber = frameNumber++;

        {
            unique_lock<mutex> lock(queueMutex);
            if (frameQueue.size() >= MAX_QUEUE_SIZE) {
                lock.unlock();
                this_thread::sleep_for(chrono::milliseconds(delay));
                continue;
            }

            frameQueue.push(move(data));
            queueCondVar.notify_one();
        }
    }

    processingDone = true;
    queueCondVar.notify_all();

    for (auto& thread : processingThreads) {
        thread.join();
    }

    destroyAllWindows();
    cap.release();

    return 0;
}
