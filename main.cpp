#include <iostream>
#include <chrono>
#include "../include/camera_calib.hpp"
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	
	//std::string pic_path = "D:/RGB_camera_calib_img";
	std::string pic_path = "./angle/all";
	//std::string pic_path = "C:/����/��ҵ���/d449b/CameraCalibration/CameraCalibration";
    CamIntrCalib cam_intr_calib(pic_path, 11, 8, 0.025);
	double t = seconds();
    cam_intr_calib.Calibrate();
	cout << "ʹ�������㷨�Ľ���ʱ�䣺" <<seconds()-t<<" s" << endl << endl;
	t = seconds();
    cam_intr_calib.CalibrateWithCv();
	cout << "ʹ��opencv�㷨�Ľ���ʱ�䣺" << seconds() - t << " s" << endl << endl;
    int a;
    cin >> a;
    return 0;
}


//int main(int argc, char** argv) {
//	//read the image
//	cv::Mat src = cv::imread("D:/ͼƬ/�����ٻ�е��ƿγ�Ա��Ӱ.jpg");  //B,G,R
//
//	if (src.empty()) {
//		printf("could not load image ");
//		return -1;
//	}
//	//��ʾͼ��
//	namedWindow("���봰��", WINDOW_FREERATIO);
//	cv::imshow("���봰��", src);
//	cv::waitKey(1000);
//	destroyAllWindows();  //��ʹ����������ᵼ����Դռ�ã�����Ӱ���������ͼ�񴰿ڡ�
//	cv::waitKey(1000);
//
//	return 0;
//}
//int main() {
//	cout << "how are you?" << endl;
//}


/* �����ļ�����ͼƬ */
//#include <iostream>  
//#include <filesystem>  
//#include <opencv2/opencv.hpp>  
//
//namespace fs = std::filesystem;
//
//int main() {
//    // ָ��Ҫ��ȡ���ļ���·��  
//    std::string folderPath = "D:/ͼƬ/˫Ŀ���ͼƬ/angle/all";
//
//    // �������ļ���  
//    for (const auto& entry : fs::directory_iterator(folderPath)) {
//        // ����ļ��Ƿ�ΪͼƬ�����Ը�����Ҫ����ļ���չ����  
//        if (entry.is_regular_file() &&
//            (entry.path().extension() == ".jpg" ||
//                entry.path().extension() == ".png" ||
//                entry.path().extension() == ".jpeg")) {
//
//            // ��ȡͼ��  
//            cv::Mat image = cv::imread(entry.path().string());
//            if (image.empty()) {
//                std::cerr << "Could not open or find the image: " << entry.path() << std::endl;
//                continue; // �����޷���ȡ��ͼ��  
//            }
//
//            // ����ͼ������ֻ����ʾ��Ϊʾ����  
//            cv::imshow("Image", image);
//            cv::waitKey(0); // �ȴ������¼�  
//        }
//    }
//
//    cv::destroyAllWindows();
//    return 0;
//}