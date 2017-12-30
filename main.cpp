#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include "Tracker.h"
using namespace std;
using namespace dlib;

void MultiTrack(string inputFilePath)
{
    cv::VideoCapture capture(inputFilePath);
    if (!capture.isOpened()) {
        std::cout << "Can't capture '' !" << std::endl;
        // return 1;
    }
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

    object_detector<image_scanner_type> detector;
    deserialize("../PedestrianDetector/model/person.svm") >> detector;
    std::vector<object_detector<image_scanner_type> > my_detectors;
    my_detectors.push_back(detector);

    image_window hogwin(draw_fhog(detector), "Learned fHOG detector");

    // correlation_tracker tracker;
    int frameCount = 0;
    int currentPersonId = 0;
    std::vector<correlation_tracker> pTrackers;
    std::vector<int> pNames;

    double scale = 2;
    for (;;) {
        cv::Mat frame, smallFrame;
        capture >> frame;
        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        cv::resize(frame, frame, cv::Size((int)(frame.cols/2.5), (int)(frame.rows/2.5)));
        cv::resize(frame, smallFrame, cv::Size((int)(frame.cols/scale), (int)(frame.rows/scale)));
        cv_image<bgr_pixel> cimg(smallFrame);

        frameCount++;
        std::vector<int> pIdToDelete;
        // cout << pTrackers.size() << endl;
        for (size_t pId = 0; pId < pTrackers.size(); ++pId) {
            double trackQuality = pTrackers[pId].update(cimg);
            if (trackQuality < 7)
                pIdToDelete.push_back(pId);
        }
        // cout << pIdToDelete.size() << endl;
        for (auto &&pID : pIdToDelete) {
            // cout <<"Removing tracker " << pID <<" from list of trackers"<<endl;
            pTrackers.erase(pTrackers.begin()+pID);
        }

        if (frameCount % 10 == 0) {
            std::vector<rectangle> dets = evaluate_detectors(my_detectors, cimg);

            for (auto &&d : dets) {
                long x = (long)d.left();
                long y = (long)d.top();
                long w = (long)d.width();
                long h = (long)d.height();

                // long maxArea = d.area();
                // center point
                long x_bar = x + 0.5*w;
                long y_bar = y + 0.5*h;

                int matchedId = -1;
                for (size_t pId = 0; pId < pTrackers.size(); ++pId) {
                    drectangle tracker_postion = pTrackers[pId].get_position();
                    long track_x = (long)tracker_postion.left();
                    long track_y = (long)tracker_postion.top();
                    long track_w = (long)tracker_postion.width();
                    long track_h = (long)tracker_postion.height();
                    long track_x_bar = track_x + 0.5*track_w;
                    long track_y_bar = track_y + 0.5*track_h;

                    if ((track_x <= x_bar) && (x_bar <= (track_x + track_w))
                        && (track_y <= y_bar) && (y_bar <= (track_y + track_h))
                        && (x <= track_x_bar) && (track_x_bar <= (x   + w))
                        && (y <= track_y_bar) && (track_y_bar <= (y   + h)))
                        matchedId = pId;
                }
                if (matchedId == -1) {
                    // currentPersonId = 0;
                    cout << "Create new tracker : " << currentPersonId << endl;
                    correlation_tracker tracker;
                    tracker.start_track(cimg, rectangle(point(x-10, y-20), point(x+w+10, y+h+20)));
                    pTrackers.push_back(tracker);
                    pNames.push_back(currentPersonId);
                    currentPersonId++;
                }
            }
        }

        for (size_t pId = 0; pId < pTrackers.size(); ++pId) {
            drectangle tracker_postion = pTrackers[pId].get_position();
            long track_x = (long)tracker_postion.left();
            long track_y = (long)tracker_postion.top();
            long track_w = (long)tracker_postion.width();
            long track_h = (long)tracker_postion.height();
            cv::Rect new_pos = cv::Rect(cvPoint(track_x*scale+10, track_y*scale),
                                        cvPoint((track_x+track_w)*scale+10,
                                                (track_y+track_h)*scale));
            cv::rectangle(frame, new_pos, cv::Scalar(0, 165, 255), 2);
            if (std::find(pNames.begin(), pNames.end(), pId) != pNames.end()) {
                cv::putText(frame, "Person " + std::to_string(
                                pId), cvPoint(track_x*scale+10, track_y*scale),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5, cv::Scalar(255, 255, 255), 2);
            } else {
                cv::putText(frame, "Detecting...",
                            cvPoint(track_x*scale+10, track_y*scale),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5, cv::Scalar(255, 255, 255), 2);
            }
        }

        cv::imshow("WINDOW_NAME", frame);
        if (cvWaitKey(1) == 27) // "press 'Esc' to break video";
            break;
    }
    cvWaitKey(0);
}

void HOG_Detect(string inputFilePath)
{
    cv::VideoCapture capture(inputFilePath);
    if (!capture.isOpened()) {
        std::cout << "Can't capture '' !" << std::endl;
        // return 1;
    }
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

    object_detector<image_scanner_type> detector;
    deserialize("../PedestrianDetector/model/person.svm") >> detector;
    std::vector<object_detector<image_scanner_type> > my_detectors;
    my_detectors.push_back(detector);
    matrix<unsigned char> tmp_img = draw_fhog(detector);
    // cout << tmp_img.nc() << "X" << tmp_img.nr() << endl;
    image_window hogwin(tmp_img, "Learned fHOG detector");

    // Now for the really fun part.  Let's display the testing images on the screen and
    // show the output of the face detector overlaid on each image.  You will see that
    // it finds all the faces without false alarming on any non-faces.
    image_window win;
    correlation_tracker tracker;
    bool isTracking = false;

    double scale = 2;
    for (;;) {
        cv::Mat frame, smallFrame;
        capture >> frame;
        cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        cv::resize(frame, frame, cv::Size((int)(frame.cols/2.5), (int)(frame.rows/2.5)));
        cv::resize(frame, smallFrame, cv::Size((int)(frame.cols/scale), (int)(frame.rows/scale)));
        cv_image<bgr_pixel> cimg(smallFrame);

        if (!isTracking) {
            std::vector<rectangle> dets = evaluate_detectors(my_detectors, cimg);
            long maxArea = 0;
            long x = 0;
            long y = 0;
            long w = 0;
            long h = 0;

            for (auto &&d : dets) {
                if ((long)d.area() > maxArea) {
                    x = (long)d.left();
                    y = (long)d.top();
                    w = (long)d.width();
                    h = (long)d.height();
                    maxArea = d.area();
                }
            }

            if (maxArea > 0) {
                tracker.start_track(cimg, rectangle(point(x, y), point(x+w, y+h)));
                isTracking = true;
            }
        } else {
            double trackQuality = tracker.update(cimg);
            if (trackQuality > 8.75) {
                drectangle pos = tracker.get_position();
                cv::Rect new_pos = cv::Rect(cvPoint(pos.left()*scale, pos.top()*scale),
                                            cvPoint(pos.right()*scale, pos.bottom()*scale));
                cv::rectangle(frame, new_pos, cv::Scalar(0, 165, 255), 1.5);
            } else {
                isTracking = false;
            }
        }

        cv::imshow("WINDOW_NAME", frame);
        if (cvWaitKey(10) == 27) // "press 'Esc' to break video";
            break;
    }
    cvWaitKey(0);
}

int main()
{
     string inputFilePath = "../PedestrianDetector/222.mp4";
    //string inputFilePath = "rtsp://10.12.11.222:554/av0_0";
    MultiTrack(inputFilePath);
}
