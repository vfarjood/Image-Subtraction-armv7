#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>


struct Centroid 
{
    int id;
    float conf;
    cv::Rect box;
    cv::Point center;
    std::string name;
};


class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
    std::chrono::duration<float> duration;

    void start() {begin = std::chrono::steady_clock::now();}

    float stop()
    {
        end = std::chrono::steady_clock::now();
        duration = end - begin; 
        float second = duration.count();
        return second;
    }

};



int main(int argc, char** argv )
{
    Timer total_time;
    total_time.start();

    Timer time;
    time.start();

    std::vector<Centroid> centroids;
    centroids.reserve(10);

    // Load class names:
    std::vector<std::string> class_names;
    class_names.reserve(100);
    // std::ifstream ifs(std::string("../model/object_detection_classes_coco.txt").c_str());
    // std::string line;
    // while (getline(ifs, line)) {class_names.emplace_back(line);} 
    
    // // load the neural network model:
    // cv::dnn::Net model = cv::dnn::readNet("../model/frozen_inference_graph.pb",
    //                               "../model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", 
    //                               "TensorFlow");

    // Load the media:
    cv::Mat background = cv::imread("../media/traffic/image-0.jpg", 1);
    cv::Mat image = cv::imread("../media/traffic/image-1.jpg", 1);
    std::cout << "Loading Time: " << time.stop() << "\n";

    // create result file:
    std::ofstream result_file ("result.txt");
    if (!result_file.is_open())
    {
        std::cout << "Unable to open file";
    }


    time.start();

    cv::Mat imgDifference;
    cv::Mat imgSubtracted;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> convexHulls;

    cv::cvtColor(background, background, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(background, background, cv::Size(5, 5), 0);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0);

    cv::absdiff(background, image, imgDifference);

    cv::threshold(imgDifference, imgSubtracted, 30, 255.0, cv::THRESH_BINARY);

    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (unsigned int i = 0; i < 2; i++) 
    {
        cv::dilate(imgSubtracted, imgSubtracted, structuringElement5x5);
        cv::dilate(imgSubtracted, imgSubtracted, structuringElement5x5);
        cv::erode (imgSubtracted, imgSubtracted, structuringElement5x5);
    }

    cv::findContours(imgSubtracted, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > Hulls(contours.size());
    for (unsigned int i = 0; i < contours.size(); i++) 
    {
        cv::convexHull(contours[i], Hulls[i]);
        convexHulls.emplace_back(Hulls[i]);

    }

    int carID = 0;
    for (auto &convexHull : convexHulls) 
    {
        // Centroid possibleVehicle(convexHull);
        std::vector<cv::Point> contour = convexHull;
        Centroid object;

        object.id         = carID;
        object.name       = "car ";
        object.conf       = 0.0;
        object.box = cv::boundingRect(contour);
        object.center     = cv::Point((object.box.x + object.box.width)/2, (object.box.y + object.box.height)/2);

        if (object.box.area() > 400 && 
                object.box.width > 30 && 
                    object.box.height > 30 &&
                        (cv::contourArea(contour) / (double)object.box.area()) > 0.50) 
        {
            centroids.emplace_back(object);
            carID++;
        }
    }

    std::cout << "Computation Time: " << time.stop() << "\n";
    std::cout << "Total Time: " << total_time.stop() << "\n";
    std::cout << "Number of detected cars: " << centroids.size() << "\n";

    for(int i=0; i < centroids.size(); i++)
    {
        result_file << "name: "  << centroids[i].name << "\n";
        result_file << "ID: "    << centroids[i].id << "\n";
        result_file << "conf: "  << centroids[i].conf << "\n";
        result_file << "center: "<< centroids[i].center << "\n";
        result_file << "x: "     << centroids[i].box.x << "\n";
        result_file << "y: "     << centroids[i].box.y << "\n";
        result_file << "w: "     << centroids[i].box.width  << "\n";
        result_file << "h: "     << centroids[i].box.height << "\n";
        result_file << "---------------------" << "\n";
    }

    return 0;
}