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

#define CROP_X 602
#define CROP_Y 337
#define CROP_WIDTH 50
#define CROP_HEIGHT 20
#define CROP_SCALE 10

#define STABLE_X 4
#define STABLE_Y 4
#define STABLE_WIDTH 22
#define STABLE_HEIGHT 22

#define MEDIAN_SIZE 5
#define ERODE_SIZE 3
#define DILATE_SIZE 3

#define FILE_FPS 25
#define PREVIEW_DELAY 40

#define CROP_FILE "01_crop.avi"
#define STABLE_FILE "02_stable.avi"
#define SUPER_FILE "03_super.avi"
#define FINAL_FILE "04_final.avi"

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
    
    cvtColor( crop, crop, CV_BGR2GRAY );
    equalizeHist( crop, crop );
    cvtColor( crop, crop, CV_GRAY2BGR );
    
    writer << crop;
    
    resize( crop, crop, CROP_SIZE );
    
    imshow( "Tracking", crop );
    
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
//     resize( frame, frame, CROP_SIZE );
    imshow( "Frame", frame );
    
    tracker->init(frame, roi);
    
    for( int i = 0;; ++i ) {
        if( !cap.read(frame) ) break;
        
        tracker->update(frame, roi);
        
        crop = frame(roi);
        
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        cvtColor( crop, crop, CV_BGR2GRAY );
        threshold( crop, crop, 50, 255, 3 );
        equalizeHist( crop, crop );
        cvtColor( crop, crop, CV_GRAY2BGR );
        
        resize( crop, crop, FILE_SIZE, 0, 0, INTER_LANCZOS4 );
        writer << crop;
        
        resize( crop, crop, CROP_SIZE );
        imshow( "Tracking", crop );
        
        rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
//         resize( frame, frame, CROP_SIZE );
        imshow("Frame", frame);
    }
    destroyWindow( "Frame" );
    destroyWindow( "Tracking" );
}

void stabilize( string inFile, string outFile ) {
    
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    
    Mat fA;
    Rect roi( STABLE_X, STABLE_Y, STABLE_WIDTH, STABLE_HEIGHT );
    
    VideoWriter writer;
    writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, FILE_SIZE );
    if( !writer.isOpened() ) {
        cerr << "Output file cannot be opened" << endl;
        return;
    } else {
        if( !cap.read(fA) ) return;
//         medianBlur( fA, fA, 4 );
//         Sobel( fA, fA, CV_8U, 0, 1 );
//         erode( fA, fA, getStructuringElement( MORPH_ELLIPSE, Size(1,3)) );
//         dilate( fA, fA, getStructuringElement( MORPH_ELLIPSE, Size(1,2)) );
        
//         writer << fA;
        cvtColor( fA, fA, CV_BGR2GRAY );
//         Canny( fA, fA, 128, 255, 3);
    }
    
    for( int i = 0;; ++i ) {
        Mat fB;
        if( !cap.read(fB) ) break;
//         if( i%13!=0 ) continue;
        cout << '[' << setw(3) << i << "] : " << flush;
        
        TickMeter tm;
        tm.start();
        
//         medianBlur( fB, fB, 4 );
        cvtColor(fB, fB, CV_BGR2GRAY);
//         Canny( fB, fB, 50, 150, 3);
//         Sobel( fB, fB, CV_8U, 0, 1 );
//         erode( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(1,3)) );
//         dilate( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(1,2)) );
        
        
//         if( i%100 == 0) fA = fB;
        Mat warp_matrix = Mat::eye(2, 3, CV_32F);
        int number_of_iterations = 100;
        double termination_eps = 1e-10;
        TermCriteria criteria( TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps );
        findTransformECC( fB, fA, warp_matrix, MOTION_AFFINE, criteria );
//         warp_matrix = estimateRigidTransform(fB, fA, false);
        warpAffine( fB, fB, warp_matrix, fB.size(), INTER_LANCZOS4 );
//         warpPerspective( fB, fB, warp_matrix, fB.size()/*, INTER_LINEAR + WARP_INVERSE_MAP */);
        
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        threshold( fB, fB, 80, 255, 3 );
        equalizeHist( fB, fB );
//         fB = fB( roi );
        cvtColor( fB, fB, CV_GRAY2BGR);
        writer << fB;
        
        resize( fB, fB, CROP_SIZE );
        imshow("Stabilization", fB );
    }
    destroyWindow( "Stabilization" );
    
}

void superRes( string inFile, string outFile ) {
    
    static const int scale = 10;
    static const int iterations = 10;
    static const int temporalAreaRadius = 50;
    
    Ptr<FrameSource> frameSource;
    
    cout << "Opening file " << inFile << endl;
    frameSource = createFrameSource_Video( inFile );
    
//     Mat frame;
//     frameSource->nextFrame(frame);
//     frameSource->nextFrame(frame);
//     frameSource->nextFrame(frame);
    
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
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        if( !writer.isOpened() ) 
            writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, result.size() );
        writer << result;
        
        resize( result, result, CROP_SIZE );
        
        imshow("Super Resolution", result);
    }
    destroyWindow( "Super Resolution" );
    
}

void morph( string inFile, string outFile ) {
    
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

        char d = 8, s = -1, c = -1;
        char kernel_data[] = {
             c,s,c, 
             s,d,s, 
             c,s,c
        };
        Mat kernel( 3, 3, CV_8S, kernel_data );
        
//         erode( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
//         dilate( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
//         dilate( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
//         erode( fB, fB, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
        morphologyEx( fB, fB, MORPH_GRADIENT, getStructuringElement( MORPH_RECT, Size(4,4)));
//         threshold(fB, fB, 60, 255, 3);
        bitwise_not( fB, fB );
        medianBlur( fB, fB, 9 );
        equalizeHist( fB, fB );
        threshold(fB, fB, 0, 255, THRESH_BINARY | THRESH_OTSU);
        filter2D( fB, fB, -1, kernel );
        
        fA = fB;
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        cvtColor( fB, fB, CV_GRAY2BGR);
        writer << fB;
        
        resize( fB, fB, CROP_SIZE );
        imshow("Morph", fB );
    }
    destroyWindow( "Morph" );
    
}


int main( int argc, const char* argv[] ) {

    if( argc < 1 ) {
        cout << "Missing input file" << endl;
        return 1;
    }
    
//     track( argv[1], CROP_FILE );
//     stabilize( CROP_FILE, STABLE_FILE );
//     superRes( CROP_FILE, SUPER_FILE );
    morph( SUPER_FILE, FINAL_FILE );
    
    return 0;
}
