#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

#define CROP_X 527/*146*/
#define CROP_Y 375/*116*/
#define CROP_WIDTH 60
#define CROP_HEIGHT 60
#define CROP_SCALE 10

#define MEDIAN_SIZE 3
#define ERODE_SIZE 4
#define DILATE_SIZE 4

#define FILE_FPS 25

#define CROP_FILE "crop.avi"
#define STABLE_FILE "stable.avi"
#define FINAL_FILE "final.avi"

#define FILE_SIZE Size(CROP_WIDTH, CROP_HEIGHT)
#define CROP_SIZE Size(CROP_WIDTH*CROP_SCALE, CROP_HEIGHT*CROP_SCALE)

void track( string inFile, string outFile ) {
    
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    VideoWriter writer;
    writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, FILE_SIZE );
    if( !writer.isOpened() ) {
        cerr << "Output file cannot be opened" << endl;
        return;
    }
    
//     Ptr<Tracker> tracker = TrackerBoosting::create();
//     Ptr<Tracker> tracker = TrackerMIL::create();
    Ptr<Tracker> tracker = TrackerKCF::create();
//     Ptr<Tracker> tracker = TrackerTLD::create();
//     Ptr<Tracker> tracker = TrackerMedianFlow::create();
//     Ptr<Tracker> tracker = TrackerGOTURN::create();
    
    Mat frame, crop;
    cap.read(frame);
    
    Rect2d roi(CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT);
    
    crop = frame(roi);
    
    cvtColor(crop, crop, CV_BGR2GRAY);
    equalizeHist( crop, crop );
    cvtColor(crop, crop, CV_GRAY2BGR);
    writer << crop;
    
    resize( crop, crop, CROP_SIZE );
    
    imshow("Tracking", crop);
    
    rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    resize( frame, frame, CROP_SIZE );
    imshow("Frame", frame);
    
    tracker->init(frame, roi);
    
    for( int i = 0;; ++i ) {
        if( !cap.read(frame) ) break;
        
        tracker->update(frame, roi);
        
        crop = frame(roi);
        
        if( waitKey(50) > 0 )  break;
        
        cvtColor(crop, crop, CV_BGR2GRAY);
        equalizeHist( crop, crop );
        cvtColor(crop, crop, CV_GRAY2BGR);
        writer << crop;
        
        resize( crop, crop, CROP_SIZE );
        imshow("Tracking", crop);
        
        rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
        resize( frame, frame, CROP_SIZE );
        imshow("Frame", frame);
    }
}

void stabilize( string inFile, string outFile ) {
    
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    
    Mat fA;
    
    VideoWriter writer;
    writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, FILE_SIZE );
    if( !writer.isOpened() ) {
        cerr << "Output file cannot be opened" << endl;
        return;
    } else {
        if( !cap.read(fA) ) return;
        writer << fA;
        cvtColor( fA, fA, CV_BGR2GRAY );
    }
    
    for( int i = 0;; ++i ) {
        Mat fB;
        if( !cap.read(fB) ) break;
        cout << '[' << setw(3) << i << "] : " << flush;
        
        TickMeter tm;
        tm.start();
        
        cvtColor(fB, fB, CV_BGR2GRAY);
        
//         Mat warp_matrix = Mat::eye(2, 3, CV_32F);
//         int number_of_iterations = 50;
//         double termination_eps = 1e-10;
//         TermCriteria criteria( TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps );
//         findTransformECC( fA, fB, warp_matrix, MOTION_TRANSLATION, criteria );
//         warpAffine( fB, fB, warp_matrix, fB.size(), INTER_LINEAR + WARP_INVERSE_MAP) ;
//         warpPerspective( fB, fB, warp_matrix, fB.size(),INTER_LINEAR + WARP_INVERSE_MAP );
                
//         medianBlur( fB, fB, MEDIAN_SIZE );
//         erode( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
//         dilate( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
//         dilate( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
//         erode( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
        
        equalizeHist( fB, fB );
        threshold(fB, fB, 190, 255, 0);
        
        fA = fB;
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        if( waitKey(40) > 0 )  break;
        
        cvtColor( fB, fB, CV_GRAY2BGR);
        writer << fB;
        
        resize( fB, fB, CROP_SIZE );
        imshow("Stabilization", fB );
    }
    
}

void superRes( string inFile, string outFile ) {
    
    static const int scale = 4;
    static const int iterations = 10;
    static const int temporalAreaRadius = 20;
    
    Ptr<FrameSource> frameSource;
    
    cout << "Opening file " << inFile << endl;
    frameSource = createFrameSource_Video( inFile );
    
    Mat frame;
    frameSource->nextFrame(frame);
    
    Ptr<SuperResolution> superRes;   
    superRes = createSuperResolution_BTVL1();
    superRes->setOpticalFlow( cv::superres::createOptFlow_DualTVL1() );
    superRes->setScale( scale );
    superRes->setIterations( iterations );
    superRes->setTemporalAreaRadius( temporalAreaRadius );
    superRes->setInput( frameSource );

    VideoWriter writer;

    for( int i = 0;; ++i ) {
        cout << '[' << setw(3) << i << "] : " << flush;
        Mat result;
        
        TickMeter tm;
        tm.start();
        superRes->nextFrame( result );
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        if( result.empty() ) break;
        if( waitKey(100) > 0 )  break;
        
        if( !writer.isOpened() ) 
            writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, result.size() );
        writer << result;
        
        resize( result, result, CROP_SIZE );
        
        imshow("Super Resolution", result);
    }
    
}

int main( int argc, const char* argv[] ) {

    if( argc < 1 ) {
        cout << "Missing input file" << endl;
        return 1;
    }
    
    track( argv[1], CROP_FILE );
    superRes( CROP_FILE, STABLE_FILE );
    stabilize( STABLE_FILE, FINAL_FILE );
    
    return 0;
}
