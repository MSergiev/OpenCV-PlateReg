#include <iostream>
#include <iomanip>
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
#include "opencv2/photo.hpp"

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
#define ERODE_SIZE 2
#define DILATE_SIZE 2

#define FILE_FPS 25
#define PREVIEW_DELAY 40

#define INPUT_FILE "00_input.mp4"
#define CROP_FILE "01_crop.avi"
#define STABLE_FILE "02_stable.avi"
#define SUPER_FILE "03_super.avi"
#define FINAL_FILE "04_final.avi"

#define FILE_SIZE Size(CROP_WIDTH, CROP_HEIGHT)
#define CROP_SIZE Size(CROP_WIDTH*CROP_SCALE, CROP_HEIGHT*CROP_SCALE)




// Function prototypes
void track( string inFile, string outFile );
void stabilize( string inFile, string outFile );
void superRes( string inFile, string outFile );
void morph( string inFile, string outFile );




// Main method
int main( int argc, const char* argv[] ) {

    if( argc != 2 ) {
        cout << "Missing options" << endl;
        cout << "-t to track" << endl;
        cout << "-s to stabilize" << endl;
        cout << "-r to superres" << endl;
        cout << "-m to morph" << endl;
        return 1;
    }
    
    switch( argv[1][1] ) {
        case 't': track( INPUT_FILE, CROP_FILE ); break;
        case 's': stabilize( CROP_FILE, STABLE_FILE ); break;
        case 'r': superRes( CROP_FILE, SUPER_FILE ); break;
        case 'm': morph( SUPER_FILE, FINAL_FILE ); break;
        default: cout << "Invalid argument" << endl;
    }
    
    return 0;
}




// Tracking function
void track( string inFile, string outFile ) {
    
    // Open input file
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    // Open output file
    VideoWriter writer;
    writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, FILE_SIZE );
    if( !writer.isOpened() ) {
        cerr << "Output file cannot be opened" << endl;
        return;
    }
    
    // Select tracking algorithm
//     Ptr<Tracker> tracker = TrackerBoosting::create();
//     Ptr<Tracker> tracker = TrackerMIL::create();
    Ptr<Tracker> tracker = TrackerKCF::create();
//     Ptr<Tracker> tracker = TrackerTLD::create();
//     Ptr<Tracker> tracker = TrackerMedianFlow::create();
//     Ptr<Tracker> tracker = TrackerGOTURN::create();
    
    // Read first frame
    Mat frame, crop;
    cap.read(frame);
    
    // Define region of interest
    Rect2d roi(CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT);
    
    // Crop frame to ROI rectangle
    crop = frame(roi);
    
    // Equalize grayscale histogram
    cvtColor( crop, crop, CV_BGR2GRAY );
    equalizeHist( crop, crop );
    cvtColor( crop, crop, CV_GRAY2BGR );
    
    // Write cropped file to file
    writer << crop;
    
    // Resize cropped file for the preview
    resize( crop, crop, CROP_SIZE );
    
    // Show cropped image
    imshow( "Tracking", crop );
    
    // Draw ROI on the frame
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    
    // Show frame
    imshow( "Frame", frame );
    
    // Initialize tracker
    tracker->init(frame, roi);
    
    // Process video frames
    for( int i = 0;; ++i ) {
        // Read frame
        if( !cap.read(frame) ) break;
        
        // Update ROI with new frame data
        tracker->update(frame, roi);
        
        // Crop frame to ROI rectangle
        crop = frame(roi);
        
        // Check for keypress
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        // Threshold image to increase contrast
        cvtColor( crop, crop, CV_BGR2GRAY );
        threshold( crop, crop, 50, 255, THRESH_TOZERO | THRESH_OTSU );
        equalizeHist( crop, crop );
        cvtColor( crop, crop, CV_GRAY2BGR );
        
        // Resize cropped image
        resize( crop, crop, FILE_SIZE, 0, 0, INTER_LANCZOS4 );
        
        // Write cropped image to file
        writer << crop;
        
        // Resize cropped image for the preview
        resize( crop, crop, CROP_SIZE );
        
        // Show cropped image
        imshow( "Tracking", crop );
        
        // Draw ROI on frame
        rectangle(frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
        
        // Show frame
        imshow("Frame", frame);
    }
    
    // Release windows
    destroyWindow( "Frame" );
    destroyWindow( "Tracking" );
}




// Stabilizing function
void stabilize( string inFile, string outFile ) {
    
    // Open input file
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    
    // Create comparison frame matrix
    Mat fA;
    
    // Define stabilization region of interest
    Rect roi( STABLE_X, STABLE_Y, STABLE_WIDTH, STABLE_HEIGHT );
    
    // Video writer instance
    VideoWriter writer;
    
    // Read comparison frame
    if( !cap.read(fA) ) return;
    cvtColor( fA, fA, CV_BGR2GRAY );
        
    // Process video frames
    for( int i = 0;; ++i ) {
        // Read current frame
        Mat fB;
        if( !cap.read(fB) ) break;
        
        // Draw current frame index
        cout << '[' << setw(3) << i << "] : " << flush;
        
        // Start processing timer
        TickMeter tm;
        tm.start();
        
        // Convert to grayscale
        cvtColor(fB, fB, CV_BGR2GRAY);
        
        // Create affine transformation matrix
        Mat warp_matrix = Mat::eye(2, 3, CV_32F);
        
        // Number of processing iterations
        const int number_of_iterations = 100;
        
        // Minimum transformation offset
        const double termination_eps = 1e-10;
        
        // Create transfromation termination criterion
        TermCriteria criteria( TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps );
        
        // Find affine transformation from fA to fB
        findTransformECC( fB, fA, warp_matrix, MOTION_AFFINE, criteria );
        
        // Warp fB according to the found transformation
        warpAffine( fB, fB, warp_matrix, fB.size(), INTER_LANCZOS4 );
        
        // Stop and print timer
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        // Check for keypress
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        // Convert to RGB
        cvtColor( fB, fB, CV_GRAY2BGR );
        
        // Write to file
        if( !writer.isOpened() ) 
            writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, fB.size() );
        writer << fB;
        
        // Resize frame for preview
        resize( fB, fB, CROP_SIZE );
        
        // Show stabilized frame
        imshow( "Stabilization", fB );
    }
    
    // Release window
    destroyWindow( "Stabilization" );
    
}




// Super resolution function
void superRes( string inFile, string outFile ) {
    
    // Superres scale factor
    static const int scale = 10;
    // Superres iteration count
    static const int iterations = 10;
    // Superres frame radius
    static const int temporalAreaRadius = 40;
    
    // Create framesource
    Ptr<FrameSource> frameSource;
    
    cout << "Opening file " << inFile << endl;
    frameSource = createFrameSource_Video( inFile );
    
    // Skip first frame
    {
        Mat frame;
        frameSource->nextFrame(frame);
    }
        
    // Create superres instance
    Ptr<SuperResolution> superRes;
    // Use BTVL1 algorithm
    superRes = createSuperResolution_BTVL1();
    // Use DualTVL1 optical flow algorithm
    superRes->setOpticalFlow( cv::superres::createOptFlow_DualTVL1() );
    // Set parameters
    superRes->setScale( scale );
    superRes->setIterations( iterations );
    superRes->setTemporalAreaRadius( temporalAreaRadius );
    superRes->setInput( frameSource );

    // Video writer instance
    VideoWriter writer;

    // Process frames
    for( int i = 0;; ++i ) {
        // Print current frame index
        cout << '[' << setw(3) << i << "] : " << flush;
        
        // Create result image matrix
        Mat result;
        
        // Start processing timer
        TickMeter tm;
        tm.start();
        
        // Process frame
        superRes->nextFrame( result );
        
        // Stop and print timer
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        // Check for invalid result
        if( result.empty() ) break;
        
        // Check for keypress
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        // Write result frame to file
        if( !writer.isOpened() ) 
            writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, result.size() );
        writer << result;
        
        // Resize frame for the preview
        resize( result, result, CROP_SIZE );
        
        // Show frame
        imshow("Super Resolution", result);
    }
    
    // Release window
    destroyWindow( "Super Resolution" );
    
}




// Morphing function
void morph( string inFile, string outFile ) {
    
    // Open input file
    VideoCapture cap( inFile );
    if( !cap.isOpened() ) {
        cerr << "Input file cannot be opened" << endl;
        return;
    }
    
    // Video writer instance
    VideoWriter writer;
    
    // Create frame matrix
    Mat frame;
    
    // Convolution kernel definition
    char d = 5, s = -1, c = 0;
    char kernel_data[] = {
            c,s,c, 
            s,d,s, 
            c,s,c
    };
    Mat kernel( 3, 3, CV_8S, kernel_data );
    
    // Process frames
    for( int i = 0;; ++i ) {
        // Read frame
        if( !cap.read(frame) ) break;
        
        // Print current frame index
        cout << '[' << setw(3) << i << "] : " << flush;
        
        // Start processing timer
        TickMeter tm;
        tm.start();
        
        // Convert to grayscale
        cvtColor(frame, frame, CV_BGR2GRAY);

        // Apply filtering operations
//         morphologyEx( frame, frame, MORPH_GRADIENT, getStructuringElement( MORPH_RECT, Size(4,4)));
//         threshold(frame, frame, 60, 255, 3);
//         bitwise_not( frame, frame );
//         medianBlur( frame, frame, 15 );
//         equalizeHist( frame, frame );
//         threshold(frame, frame, 180, 255, THRESH_BINARY | THRESH_OTSU);
        for( unsigned char i = 0; i < 1; ++i ) {
//             filter2D( frame, frame, -1, kernel );
//             erode( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
//             dilate( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
//             medianBlur( frame, frame, 3 );
            dilate( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
            erode( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
        }
//         fastNlMeansDenoising( frame, frame, 5, 30 );
//         equalizeHist( frame, frame );
//         threshold(frame, frame, 180, 255, THRESH_BINARY);
        adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 2);
//         dilate( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(ERODE_SIZE,ERODE_SIZE)) );
//         erode( frame, frame, getStructuringElement( MORPH_ELLIPSE, Size(DILATE_SIZE,DILATE_SIZE)) );
        
        // Stop processing timer
        tm.stop();
        cout << tm.getTimeSec() << " sec" << endl;
        
        // Wait for keypress
        if( waitKey(PREVIEW_DELAY) > 0 )  break;
        
        // Convert to RGB
        cvtColor( frame, frame, CV_GRAY2BGR);
        
        // Write frame to file        
        if( !writer.isOpened() ) 
            writer.open( outFile, CV_FOURCC('X','V','I','D'), FILE_FPS, frame.size() );
        writer << frame;
        
        // Resize frame for the preview
        resize( frame, frame, CROP_SIZE );
        
        // Show frame
        imshow("Morph", frame );
    }
    
    // Release window
    destroyWindow( "Morph" );
    
}
