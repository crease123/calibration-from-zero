#include <iostream>
#include <chrono>
#include "../include/camera_calib.hpp"
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	
	//std::string pic_path = "D:/RGB_camera_calib_img";
	std::string pic_path = "./angle/all";
	//std::string pic_path = "C:/桌面/毕业设计/d449b/CameraCalibration/CameraCalibration";
    CamIntrCalib cam_intr_calib(pic_path, 11, 8, 0.025);
	double t = seconds();
    cam_intr_calib.Calibrate();
	cout << "使用自研算法的解算时间：" <<seconds()-t<<" s" << endl << endl;
	t = seconds();
    cam_intr_calib.CalibrateWithCv();
	cout << "使用opencv算法的解算时间：" << seconds() - t << " s" << endl << endl;
    int a;
    cin >> a;
    return 0;
}


//int main(int argc, char** argv) {
//	//read the image
//	cv::Mat src = cv::imread("D:/图片/大二暑假机械设计课成员合影.jpg");  //B,G,R
//
//	if (src.empty()) {
//		printf("could not load image ");
//		return -1;
//	}
//	//显示图像
//	namedWindow("输入窗口", WINDOW_FREERATIO);
//	cv::imshow("输入窗口", src);
//	cv::waitKey(1000);
//	destroyAllWindows();  //不使用这个函数会导致资源占用，甚至影响后续简历图像窗口。
//	cv::waitKey(1000);
//
//	return 0;
//}
//int main() {
//	cout << "how are you?" << endl;
//}


/* 遍历文件夹中图片 */
//#include <iostream>  
//#include <filesystem>  
//#include <opencv2/opencv.hpp>  
//
//namespace fs = std::filesystem;
//
//int main() {
//    // 指定要读取的文件夹路径  
//    std::string folderPath = "D:/图片/双目相机图片/angle/all";
//
//    // 遍历该文件夹  
//    for (const auto& entry : fs::directory_iterator(folderPath)) {
//        // 检查文件是否为图片（可以根据需要检查文件扩展名）  
//        if (entry.is_regular_file() &&
//            (entry.path().extension() == ".jpg" ||
//                entry.path().extension() == ".png" ||
//                entry.path().extension() == ".jpeg")) {
//
//            // 读取图像  
//            cv::Mat image = cv::imread(entry.path().string());
//            if (image.empty()) {
//                std::cerr << "Could not open or find the image: " << entry.path() << std::endl;
//                continue; // 跳过无法读取的图像  
//            }
//
//            // 处理图像（这里只是显示作为示例）  
//            cv::imshow("Image", image);
//            cv::waitKey(0); // 等待按键事件  
//        }
//    }
//
//    cv::destroyAllWindows();
//    return 0;
//}