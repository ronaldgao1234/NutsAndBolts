#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"


using namespace std;
using namespace cv;

static Scalar randomColor(RNG& rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

Mat img = imread("nb.png");

Mat removeLight(Mat img, Mat pattern, int method) {
	Mat aux;
	//if method is normalization 
	if (method == 1) {
		//Require change our image to 32 float for division
		Mat img32, pattern32;
		img.convertTo(img32, CV_32F);
		pattern.convertTo(pattern32, CV_32F);
		//divide the image by the pattern
		aux = 1 - (img32 / pattern32);
		//scale it to conver tto 8bit format
		aux = aux * 255;
		aux.convertTo(aux, CV_8U);
	}else{
		aux = pattern - img;
	}
	return aux;
}

Mat calculateLightPattern(Mat img) {
	Mat pattern;
	//Basic and effective way to calculate the light pattern from one image
	blur(img, pattern, Size(img.cols / 3, img.cols / 3));
	return pattern;
}

Mat thresholdImage(Mat img_no_light, int method_light) {
	//Binarize image for segment
	Mat img_thr;
	if (method_light != 2) {
		threshold(img_no_light, img_thr, 30, 255, THRESH_BINARY);
	}else{
		threshold(img_no_light, img_thr, 140, 255, THRESH_BINARY_INV);
	}
	return img_no_light;
}

void ConnectedComponents(Mat img1) {
	//use connected components to divide our possible parts of images
	Mat labels;
	int num_objects = connectedComponents(img1, labels);
	
	//check the number of objects detected
	if (num_objects < 2) {
		cout << "no objects detected" << endl;
		return;
	}
	else {
		//num_objects -1 because of background
		cout << "Number of objects deteced: " << num_objects - 1 << endl;
	}

	//Create output image coloring the objects
	//create a new black image to contain the new objects
	Mat output = Mat::zeros(img1.rows, img1.cols, CV_8UC3);
	RNG rng(0xFFFFFFFF);
	namedWindow("Result");
	//6th one in the labels makes no sense. it literally isn't an object
	//start at 1 since we don't need the first background object
	for (int i = 1; i < num_objects; i++) {
		//u make the mask. if labels == i at some point in the image, mask will equal to 1 there. 
		Mat mask = labels == i;
		output.setTo(randomColor(rng), mask);
	}
	imshow("Result", output);
}

void ConnectedComponentsStats(Mat img) {
	namedWindow("Result");
	//Use connected components with stats
	Mat labels, stats, centroids;
	int num_objects = connectedComponentsWithStats(img, labels, stats, centroids);
	//check number of objects detected
	if (num_objects < 2) {
		cout << "no objects detected" << endl;
		return;
	}
	else {
		cout << "Number of objects detected: " << num_objects - 1 << endl;
	}

	//Create output image coloring the objects and show area
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
	RNG rng(0xFFFFFFFF);

	for (int i = 1; i < num_objects; i++) {
		//inside centroids: doubles, CV_64F
		cout << "Object " << i << "with pos: x: " << centroids.at<double>(i, 0) << " with pos: y: " 
			<< centroids.at<double>(i, 1) <<" with area " << stats.at<int>(i, CC_STAT_AREA) << endl;
		Mat mask = labels == i;
		output.setTo(randomColor(rng), mask);
		//draw text with area
		stringstream ss;
		ss << "area: " << stats.at<int>(i, CC_STAT_AREA);

		putText(output, ss.str(), Point(centroids.at<double>(i,0),centroids.at<double>(i,1)), 
			FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
	}
	imshow("Result", output);
}

void FindContoursBasic(Mat img) {
	vector<vector<Point>> contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);

	//check the number of objects detected
	if (contours.size() == 0) {
		cout << "No objects detected" << endl;
		return;
	}
	else {
		cout << "Number of objects detected: " << contours.size() << endl;
	}
	RNG rng(0xFFFFFFFF);
	for (int i = 0; i < contours.size(); i++) {
		drawContours(output, contours, i, randomColor(rng));
	}
	imshow("Result", output);
}

int main() {
	Mat pattern, img2, result;
	pattern = calculateLightPattern(img);
	//blur(img, pattern, Size(300, 300));
	//Mat structElement = getStructuringElement(MORPH_RECT, Size(5, 5));
	//erode(img2, img2, structElement);
	img2 = removeLight(img, pattern, 2);
	threshold(img2, img2, 50, 255, THRESH_BINARY);
	//img2.convertTo(img2, CV_8UC1);
	cvtColor(img2, img2, CV_BGR2GRAY);
	
	//cout << img2.type() << endl; //CV_8UC3 ==  2<<3 = 16 
	//ConnectedComponentsStats(img2);
	FindContoursBasic(img2);
	//result = thresholdImage(img2, 1);
	namedWindow("result");
	moveWindow("result", 6, 0);
	imshow("result", img2);
	waitKey(0);
}