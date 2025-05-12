#include "../include/camera_calib.hpp"
#include <filesystem> 
namespace fs = std::filesystem;
// #define DEBUG_CODE
#define USE_CV


//@func
//  这是类的构造函数，用于初始化类的实例。
//@para
//  const std::string &pic_path：图片路径，可能包含用于标定的棋盘格图像的文件夹路径。
//  const int points_per_row：棋盘格每行的角点数。
//  const int points_per_col：棋盘格每列的角点数。
//  const double square_size：棋盘格每个格子的物理尺寸，通常以毫米或米为单位。
//
//构造函数使用初始化列表来初始化类的成员变量：
//pic_path_(pic_path)：将传入的 pic_path 赋值给类的成员变量 pic_path_。

double seconds()
{
    // 获取当前系统时间点
    auto now = std::chrono::system_clock::now();

    // 将时间点转换为自1970年1月1日以来的微秒数
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    // 输出结果
    return (double)microseconds.count() / 1000000;

    //return (double)clock()/CLOCKS_PER_SEC;
}

CamIntrCalib::CamIntrCalib(const std::string &pic_path, const int points_per_row,
                           const int points_per_col,
                           const double square_size) : pic_path_(pic_path),
                                                       points_per_row_(points_per_row),
                                                       points_per_col_(points_per_col),
                                                       square_size_(square_size)
{
    K_ = cv::Mat::eye(3, 3, CV_64F);
    dist_coef_ = cv::Mat::zeros(4, 1, CV_64F);
}
//@func
//  使用 OpenCV 的 cv::calibrateCamera 函数进行相机标定。
bool CamIntrCalib::CalibrateWithCv()
{
    std::cout << "Calibrate with opencv" << std::endl;
    if (ReadPics() && GetKeyPoints())
    {
        std::vector<std::vector<cv::Point3f>> corner_3d_vec;
        Transform2dTo3dPts(points_3d_vec_, &corner_3d_vec);
        std::vector<cv::Mat> rvecs, tvecs;
        cv::Mat K;
        cv::Mat dist_coef;
//下面函数中两个数字的意义：CV_CALIB_FIX_TANGENT_DIST 2097152     CV_CALIB_FIX_K3  128
        cv::calibrateCamera(corner_3d_vec, points_2d_vec_, cv::Size(points_per_col_, points_per_row_),
                            K, dist_coef, rvecs, tvecs, 128 | 2097152);
        std::cout << "opencv calib K:\n"
                  << K << std::endl;
        std::cout << "opencv calib dist_coeff:\n"
                  << dist_coef << std::endl;
        double reproj_err = 0;
        int p_num = 0;
        std::vector<std::vector<cv::Point2f>> calibrated_2d_points;
        for (size_t i = 0; i < corner_3d_vec.size(); ++i)
        {
            std::vector<cv::Point2f> points_2d;
            cv::projectPoints(corner_3d_vec[i], rvecs[i], tvecs[i], K,
                              dist_coef, points_2d);
            for (size_t j = 0; j < corner_3d_vec[i].size(); ++j)
            {
                const cv::Point2f &reproj_p = points_2d[j];
                const cv::Point2f &origin_p = points_2d_vec_[i][j];
                reproj_err += sqrt((origin_p.x - reproj_p.x) * (origin_p.x - reproj_p.x) + (origin_p.y - reproj_p.y) * (origin_p.y - reproj_p.y));
                ++p_num;
            }
            calibrated_2d_points.push_back(points_2d);
        }
        reproj_err /= static_cast<double>(p_num);
        std::cout << "reproject error with opencv:" << reproj_err << std::endl;

#ifdef DEBUG_CODE
        for (size_t i = 0; i < ori_pics_.size(); ++i)
        {
            cv::drawChessboardCorners(ori_pics_[i], cv::Size(points_per_row_, points_per_col_), calibrated_2d_points[i], true);
            cv::imshow("Validate opencv ", ori_pics_[i]);
            cv::waitKey();
        }
#endif

        return true;
    }
    return false;
}

//@func
//  使用自定义方法（如张正友标定法）进行相机标定。
bool CamIntrCalib::Calibrate()
{
    std::cout << "Calibrate with Zhangzhengyou" << std::endl;
    if (ReadPics() && GetKeyPoints())
    {
        CalcH();
        // CalcHWithCV();
        // ValidateH();
        CalcK();
        CalcT();
        CalcDistCoeff();
        std::cout << "reproject error before ceres optimizing:" << std::endl;
        CalcRepjErr();
        double t = seconds();
        Optimize();
        std::cout << "使用自研算法的解算时间：" << seconds() - t << " s" << std::endl << std::endl;
        std::cout << "reproject error after ceres optimizing: " << std::endl;
        CalcRepjErr();
        return true;
    }
    return false;
}

//@func
//  读取指定路径下的图片。
bool CamIntrCalib::ReadPics()
{
    calib_pics_.clear();
    ori_pics_.clear();
    
    // 遍历该文件夹  
    for (const auto& entry : fs::directory_iterator(pic_path_)) {
        // 检查文件是否为图片（可以根据需要检查文件扩展名）  
        if (entry.is_regular_file() &&
            (entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".png" ||
                entry.path().extension() == ".jpeg" ||
                entry.path().extension() == ".bmp")) {

            // 读取图像  
            cv::Mat pic = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            

            if (pic.empty()) {
                std::cerr << "Could not open or find the image: " << entry.path() << std::endl;
                continue; // 跳过无法读取的图像  
            }
#ifdef DEBUG_CODE
            cv::imshow("origin image", pic);
            cv::waitKey(100);
#endif
            calib_pics_.push_back(pic);
            ori_pics_.push_back(pic);
        }
    }

        /* 原始代码，读取指定的四十张图片 */
//    for (int i = 0; i <= 40; ++i)
//    {
//        std::string single_picture_path = pic_path_ + "/" + std::to_string(i + 100000) + ".png";
//        cv::Mat pic = cv::imread(single_picture_path, cv::IMREAD_GRAYSCALE);
//        if (pic.empty())
//        {
//            std::cerr << "read picture failed: " << single_picture_path;
//            return false;
//        }
//#ifdef DEBUG_CODE
//        cv::imshow("origin image", pic);
//        cv::waitKey(100);
//#endif
//        calib_pics_.push_back(pic);
//        ori_pics_.push_back(pic);
//    }

    std::cout << "ReadPics succeed" << std::endl;
    return true;
}

//@func
//  检测棋盘格角点。
bool CamIntrCalib::GetKeyPoints()
{
    points_3d_vec_.clear();
    points_2d_vec_.clear();
    // world points
    // 世界坐标系的点只要是一行一行的即可，而每幅图都会有一个对应的外参

    //1. 构造世界坐标系中的每张图片的三维角点坐标
    for (size_t i = 0; i < calib_pics_.size(); ++i)
    {
        std::vector<cv::Point2f> points_3ds;
        for (int row = 0; row < points_per_row_; ++row)
        {
            for (int col = 0; col < points_per_col_; ++col)
            {
                cv::Point2f corner_p;
                corner_p.x = row * square_size_;
                corner_p.y = col * square_size_;
                points_3ds.push_back(corner_p);
            }
        }
        //将每张图片对应的三维角点坐标存储到 points_3d_vec_ 中。
        points_3d_vec_.emplace_back(std::move(points_3ds));
    }

    // detect chessboard corner
    // https://blog.csdn.net/guduruyu/article/details/69573824

    //2. 检测二维角点,即输出图像坐标系中的每张图片的角点坐标
    int test_n = 0;
    for (const auto &calib_pic : calib_pics_)
    {
        std::vector<cv::Point2f> corner_pts;
        test_n++;
        std::cout << test_n << std::endl;
        bool found_flag = cv::findChessboardCorners(
            calib_pic, cv::Size(points_per_col_, points_per_row_), corner_pts,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE); //!!! cv::Size(col,row)
        if (!found_flag)
        {
            std::cerr << "Don't found all corner points in "<<test_n<<" pic"<<std::endl;
            continue;
        }
        
        cv::TermCriteria criteria = cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

        //criteria：迭代终止条件，包括最大迭代次数和精度。
        //calib_pic：输入的灰度图像。
        //corner_pts：输入的角点坐标。
        //cv::Size(11, 11)：搜索窗口的大小。
        //cv::Size(-1, -1)：零区域的大小。
        cv::cornerSubPix(calib_pic, corner_pts, cv::Size(11, 11), cv::Size(-1, -1),
                         criteria);
        // #ifdef DEBUG_CODE
        //         // 角点绘制
        //         cv::drawChessboardCorners(calib_pic, cv::Size(points_per_row_, points_per_col_), corner_pts, found_flag);
        //         cv::imshow("chessboard corner image", calib_pic);
        //         cv::waitKey(300);
        // #endif
        points_2d_vec_.push_back(std::move(corner_pts));
    }

    std::cout << "GetKeyPoints succeed" << std::endl;
    return true;
}

//@func
//  对点集进行归一化。最后归一化的点集normed_point_vec,是均值为零，平均绝对值偏差为 1
void CamIntrCalib::Normalize(const std::vector<cv::Point2f> &point_vec,
                             std::vector<cv::Point2f> *normed_point_vec,
                             cv::Mat *norm_T)
{
    //初始化归一化变换矩阵 norm 为单位矩阵。
    *norm_T = cv::Mat::eye(3, 3, CV_64F);
   
    double mean_x = 0;       //用于存储点集的 x 坐标和 y 坐标的均值。
    double mean_y = 0;
    //将累加的 x 坐标和 y 坐标分别除以点集的大小，得到均值。
    for (const auto &p : point_vec)
    {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= point_vec.size();
    mean_y /= point_vec.size();
    
    double mean_dev_x = 0;      //用于存储点集的 x 坐标和 y 坐标的平均偏差。
    double mean_dev_y = 0;
    //遍历点集 point_vec，计算每个点的 x 坐标和 y 坐标与均值的绝对差，并累加
    for (const auto &p : point_vec)
    {
        mean_dev_x += fabs(p.x - mean_x);   //fabs() 函数可以用于 double、float 和 long double 类型的参数。如果需要计算整数的绝对值，应该使用 abs() 函数。
        mean_dev_y += fabs(p.y - mean_y);
    }
    mean_dev_x /= point_vec.size();
    mean_dev_y /= point_vec.size();

    //sx 和 sy 分别是 x 坐标和 y 坐标的归一化尺度因子，用于将点集的平均偏差归一化为 1。
    double sx = 1.0 / mean_dev_x;
    double sy = 1.0 / mean_dev_y;
    normed_point_vec->clear();
    for (const auto &p : point_vec)
    {
        cv::Point2f p_tmp;
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normed_point_vec->push_back(p_tmp);
    }

    //构建归一化变换矩阵 norm_T
    norm_T->at<double>(0, 0) = sx;
    norm_T->at<double>(0, 2) = -mean_x * sx;
    norm_T->at<double>(1, 1) = sy;
    norm_T->at<double>(1, 2) = -mean_y * sy;
}

//@func
//  计算单应性矩阵 H。
void CamIntrCalib::CalcH()
{
    //逐一求解每张图片对应的单应性矩阵，组合成 H_vec_
    for (size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        const auto &points_2d = points_2d_vec_[id];     //存储每张图片检测到的二维角点坐标。
        const auto &points_3d = points_3d_vec_[id];     //存储每张图片对应的三维角点坐标。
        std::vector<cv::Point2f> normed_points_2d;      //存储归一化后的二维点和三维点。
        std::vector<cv::Point2f> normed_points_3d;
        cv::Mat norm_T_2d;      //存储归一化变换矩阵。
        cv::Mat norm_T_3d;
        //Normalize 函数将点集归一化，最后归一化的点集normed_point_vec,是均值为零，平均绝对值偏差为 1。
        Normalize(points_2d, &normed_points_2d, &norm_T_2d);    //对二维点和三维点进行归一化处理，以提高数值稳定性。
        Normalize(points_3d, &normed_points_3d, &norm_T_3d);
        //初始化单应性矩阵 H 为单位矩阵。
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);

        //检查角点数量是否足够。
        int corner_size = normed_points_2d.size();
        if (corner_size < 4)
        {
            std::cerr << "corner size < 4";
            exit(-1);
        }

        //初始化单张图片的角点矩阵 A。
        cv::Mat A(corner_size * 2, 9, CV_64F, cv::Scalar(0));
        for (int i = 0; i < corner_size; ++i)
        {
            const auto &p_3d = normed_points_3d[i];
            const auto &p_2d = normed_points_2d[i];
            A.at<double>(2 * i, 0) = p_3d.x;
            A.at<double>(2 * i, 1) = p_3d.y;
            A.at<double>(2 * i, 2) = 1;
            A.at<double>(2 * i, 3) = 0;
            A.at<double>(2 * i, 4) = 0;
            A.at<double>(2 * i, 5) = 0;
            A.at<double>(2 * i, 6) = -p_2d.x * p_3d.x;
            A.at<double>(2 * i, 7) = -p_2d.x * p_3d.y;
            A.at<double>(2 * i, 8) = -p_2d.x;

            A.at<double>(2 * i + 1, 0) = 0;
            A.at<double>(2 * i + 1, 1) = 0;
            A.at<double>(2 * i + 1, 2) = 0;
            A.at<double>(2 * i + 1, 3) = p_3d.x;
            A.at<double>(2 * i + 1, 4) = p_3d.y;
            A.at<double>(2 * i + 1, 5) = 1;
            A.at<double>(2 * i + 1, 6) = -p_2d.y * p_3d.x;
            A.at<double>(2 * i + 1, 7) = -p_2d.y * p_3d.y;
            A.at<double>(2 * i + 1, 8) = -p_2d.y;
        }

        //计算奇异值分解 (SVD)
        cv::Mat U, W, VT;                                                    // A =UWV^T
        cv::SVD::compute(A, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        // 提取最小奇异值对应的特征向量
        H = VT.row(8).reshape(0, 3);

        //由于之前对点进行了归一化，需要将单应性矩阵 H 恢复到原始空间
        cv::Mat norm_T_2d_inv;
        cv::invert(norm_T_2d, norm_T_2d_inv);
        H = norm_T_2d_inv * H * norm_T_3d;
        H_vec_.push_back(H);
        // std::cout << "H:\n"
        //           << H << std::endl;
        // std::cout << "L2 norm: " << cv::norm(H) << std::endl;
    }
}

//@func
//  使用 OpenCV 的 cv::findHomography 计算单应性矩阵 H。使用最小均方误差或RANSAC方法。
void CamIntrCalib::CalcHWithCV()
{
    for (size_t id = 0; id < points_2d_vec_.size(); ++id)
    {
        const auto &points_2d = points_2d_vec_[id];
        const auto &points_3d = points_3d_vec_[id];

        cv::Mat H = cv::findHomography(points_3d, points_2d);
        H_vec_.push_back(H);
        // std::cout << "H:\n"
        //           << H << std::endl;
        // std::cout << "L2 norm: " << cv::norm(H) << std::endl;
    }
}

//@func
//  验证单应性矩阵 H 的准确性。
void CamIntrCalib::ValidateH()
{
    double reproject_err = 0;
    size_t img_num = points_3d_vec_.size();
    int pts_num = 0;
    std::vector<std::vector<cv::Point2f>> corner_pts_vec;
    for (size_t i = 0; i < img_num; ++i)
    {
        std::vector<cv::Point2f> corner_pts;
        const auto &H = H_vec_[i];

        for (size_t j = 0; j < points_3d_vec_[i].size(); ++j)
        {
            const auto &p_3d = points_3d_vec_[i][j];
            const auto &p_2d = points_2d_vec_[i][j];
            cv::Point2f rep_p;
            double s = H.at<double>(2, 0) * p_3d.x + H.at<double>(2, 1) * p_3d.y + H.at<double>(2, 2);
            rep_p.x = (H.at<double>(0, 0) * p_3d.x + H.at<double>(0, 1) * p_3d.y + H.at<double>(0, 2)) / s;
            rep_p.y = (H.at<double>(1, 0) * p_3d.x + H.at<double>(1, 1) * p_3d.y + H.at<double>(1, 2)) / s;
            // std::cout << "(u,v): (" << p_2d.x << ", " << p_2d.y << "); (u',v'): (" << rep_p.x << ", " << rep_p.y << ")" << std::endl;
            reproject_err += (fabs(p_2d.x - rep_p.x) + fabs(p_2d.y - rep_p.y));
            pts_num++;
            corner_pts.emplace_back(rep_p);
        }
        corner_pts_vec.push_back(corner_pts);
    }
    reproject_err /= static_cast<double>(2 * pts_num);
    std::cout << "H reproject error: " << reproject_err << std::endl;
#ifdef DEBUG_CODE

    for (size_t i = 0; i < ori_pics_.size(); ++i)
    {
        cv::drawChessboardCorners(ori_pics_[i], cv::Size(points_per_row_, points_per_col_), corner_pts_vec[i], true);
        cv::imshow("Validate H", ori_pics_[i]);
        cv::waitKey(800);
    }
#endif
}

//@func
//  从单应性矩阵 H 中恢复相机内参矩阵 K。也是用奇异值分解的方法进行求解
void CamIntrCalib::CalcK()
{
    cv::Mat A(points_2d_vec_.size() * 2, 6, CV_64F, cv::Scalar(0));
    for (size_t i = 0; i < points_2d_vec_.size(); ++i)
    {
        cv::Mat H = H_vec_[i];
        // 第1列
        double h11 = H.at<double>(0, 0);
        double h21 = H.at<double>(1, 0);
        double h31 = H.at<double>(2, 0);
        // 第2列
        double h12 = H.at<double>(0, 1);
        double h22 = H.at<double>(1, 1);
        double h32 = H.at<double>(2, 1);

        cv::Mat v11 = (cv::Mat_<double>(1, 6) << h11 * h11, h11 * h21 + h11 * h21, h21 * h21, h11 * h31 + h31 * h11, h21 * h31 + h31 * h21 + h31 * h31, h31 * h31);
        cv::Mat v12 = (cv::Mat_<double>(1, 6) << h11 * h12, h11 * h22 + h21 * h12, h21 * h22, h11 * h32 + h31 * h12, h21 * h32 + h31 * h22 + h31 * h32, h31 * h32);
        cv::Mat v22 = (cv::Mat_<double>(1, 6) << h12 * h12, h12 * h22 + h12 * h22, h22 * h22, h12 * h32 + h32 * h12, h22 * h32 + h32 * h22 + h32 * h32, h32 * h32);
        // std::cout << "v11: " << v11 << std::endl;
        //将向量 v_12,和 v_11−v_22 添加到矩阵 A
        v12.copyTo(A.row(2 * i));
        cv::Mat v_tmp = (v11 - v22);
        v_tmp.copyTo(A.row(2 * i + 1));
    }
    // std::cout << "A:\n"
    //           << A << std::endl;
   
    //计算奇异值分解 (SVD)
    cv::Mat U, W, VT;              // A =UWV^T
    cv::SVD::compute(A, W, U, VT); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
    // std::cout << "VT:\n"
    //           << VT << std::endl;
    
    //提取最小奇异值对应的特征向量
    cv::Mat B = VT.row(5);
    // std::cout << "B:\n"
    //           << B << std::endl;
    double B11 = B.at<double>(0, 0);
    double B12 = B.at<double>(0, 1);
    double B22 = B.at<double>(0, 2);
    double B13 = B.at<double>(0, 3);
    double B23 = B.at<double>(0, 4);
    double B33 = B.at<double>(0, 5);
    // std::cout << "B: " << B11 << "," << B12 << "," << B22 << "," << B13 << "," << B23 << "," << B33 << std::endl;

    //计算相机内参矩阵 K 的参数
    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);      //主点的 y 坐标。
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = sqrt(lambda / B11);                                          //x 轴方向的焦距
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));            // y 轴方向的焦距。
    double gamma = -B12 * alpha * alpha * beta / lambda;                    //x 和 y 轴的斜率。
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    std::cout << "K coeff: " << alpha << " , " << beta << " , " << gamma << " , " << lambda << " , " << u0 << " , " << v0 << std::endl;

    gamma = 0;
    //构建相机内参矩阵 K
    K_.at<double>(0, 0) = alpha;
    K_.at<double>(0, 1) = gamma;
    K_.at<double>(0, 2) = u0;
    K_.at<double>(1, 1) = beta;
    K_.at<double>(1, 2) = v0;
    std::cout << "K:\n"
              << K_ << std::endl;
}

//@func
//  从单应性矩阵 H 中恢复旋转矩阵 R 和平移向量 t。
void CamIntrCalib::CalcT()
{
    //计算相机内参矩阵 K 的逆矩阵
    cv::Mat K_inverse;
    cv::invert(K_, K_inverse);

    //遍历所有单应性矩阵 H
    for (const auto &H : H_vec_)
    {
        cv::Mat R_t = K_inverse * H;
        cv::Vec3d r1(R_t.at<double>(0, 0), R_t.at<double>(1, 0), R_t.at<double>(2, 0));
        cv::Vec3d r2(R_t.at<double>(0, 1), R_t.at<double>(1, 1), R_t.at<double>(2, 1));
        //通过向量叉积 r1×r2计算旋转矩阵的第三列 r 3。
        //这是因为旋转矩阵的列向量是正交的，第三列可以通过前两列的叉积得到
        cv::Vec3d r3 = r1.cross(r2);
        //构建旋转矩阵 Q
        cv::Mat Q = cv::Mat::eye(3, 3, CV_64F);
        Q.at<double>(0, 0) = r1(0);
        Q.at<double>(1, 0) = r1(1);
        Q.at<double>(2, 0) = r1(2);
        Q.at<double>(0, 1) = r2(0);
        Q.at<double>(1, 1) = r2(1);
        Q.at<double>(2, 1) = r2(2);
        Q.at<double>(0, 2) = r3(0);
        Q.at<double>(1, 2) = r3(1);
        Q.at<double>(2, 2) = r3(2);
        cv::Mat norm_Q;

        //使用 cv::normalize 函数对旋转矩阵 Q 进行归一化，确保其列向量是单位向量。
        cv::normalize(Q, norm_Q);
        
        //对归一化后的矩阵 Q 进行奇异值分解 (SVD),重构旋转矩阵 R,这一步是为了确保旋转矩阵的正交归一性。
        cv::Mat U, W, VT;                                                         // A =UWV^T
        cv::SVD::compute(norm_Q, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        cv::Mat R = U * VT;
        R_vec_.push_back(R);
        cv::Mat R_T;
        cv::transpose(R, R_T);
        // std::cout << "R*RT:\n"
        //           << R * R_T << std::endl;
        
        //提取平移向量 t
        cv::Mat t = cv::Mat::eye(3, 1, CV_64F);

        //提取矩阵 R_t的第三列作为平移向量 t。
        R_t.col(2).copyTo(t.col(0));
        t_vec_.push_back(t);
    }
}

//@func
//  计算畸变系数。包括径向畸变和切向畸变
// use one pic to calc
void CamIntrCalib::CalcDistCoeff()
{
    // calc ideal corner point Puv = KTPw
    std::vector<double> r2_vec;              // 用于存储每个点的平方距离的平方
    std::vector<cv::Point2f> ideal_point_vec;//用于存储理想（无畸变）的图像点坐标。

    cv::Mat R = R_vec_[0];//第一张图像的旋转矩阵 R 和平移向量 t。
    cv::Mat t = t_vec_[0];

    //取第一张图像的三维点。
    std::vector<cv::Point2f> point_3ds = points_3d_vec_[0];

    //计算理想图像点 P_uv = K·(R·P_w + t)
    for (const auto &p : point_3ds)
    {
        //将每个三维点 P转换为齐次坐标 p_3d
        cv::Mat p_3d = (cv::Mat_<double>(3, 1) << p.x, p.y, 0);
        //得到相机坐标系下的点。
        cv::Mat p_pic = R * p_3d + t;
        //转换为归一化的图像平面坐标（通过除以 z 坐标）
        p_pic.at<double>(0, 0) = p_pic.at<double>(0, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(1, 0) = p_pic.at<double>(1, 0) / p_pic.at<double>(2, 0);
        p_pic.at<double>(2, 0) = 1;
        //计算r^2，并将其存储到 r2_vec 中。
        double x = p_pic.at<double>(0, 0);
        double y = p_pic.at<double>(1, 0);
        double r2 = x * x + y * y;
        r2_vec.push_back(r2);

        //计算理想图像点,并将其存储到 ideal_point_vec 中。
        cv::Mat p_uv = K_ * p_pic;
        ideal_point_vec.emplace_back(p_uv.at<double>(0, 0), p_uv.at<double>(1, 0));
    }

    // 提取第一张图像的畸变图像点
    std::vector<cv::Point2f> dist_point_vec = points_2d_vec_[0];// points_2d_vec_ is distort uv
    double u0 = K_.at<double>(0, 2);
    double v0 = K_.at<double>(1, 2);

    //初始化矩阵 D 和向量 d，用于构建线性方程组。每个点生成两个
    // 方程（分别对应 x 和 y 坐标）因此总方程数为 2×ideal_point_vec.size()。
    // 矩阵 D 的每一行有 2 列，对应畸变系数 k1和 k2
    cv::Mat D = cv::Mat::eye(ideal_point_vec.size() * 2, 2, CV_64F);
    cv::Mat d = cv::Mat::eye(ideal_point_vec.size() * 2, 1, CV_64F);
    for (size_t i = 0; i < ideal_point_vec.size(); ++i)
    {
        double r2 = r2_vec[i];
        cv::Point2f distort_p = dist_point_vec[i];
        cv::Point2f ideal_p = ideal_point_vec[i];
        D.at<double>(2 * i, 0) = (ideal_p.x - u0) * r2;
        D.at<double>(2 * i, 1) = (ideal_p.x - u0) * r2 * r2;
        D.at<double>(2 * i + 1, 0) = (ideal_p.y - v0) * r2;
        D.at<double>(2 * i + 1, 1) = (ideal_p.y - v0) * r2 * r2;
        d.at<double>(0, 0) = distort_p.x - ideal_p.x;
        d.at<double>(1, 0) = distort_p.y - ideal_p.y;
    }
    cv::Mat DT;
    cv::transpose(D, DT);
    cv::Mat DTD_inverse;
    cv::invert(DT * D, DTD_inverse);
    dist_coef_ = DTD_inverse * DT * d;
    std::cout << "distort coeff: " << dist_coef_.at<double>(0, 0) << ", " << dist_coef_.at<double>(1, 0) << std::endl;
}

//@func
//  定义了一个名为 ReprojErr 的结构体，用于优化重投影误差（Reprojection Error）。
// 重投影误差是计算机视觉中相机标定和三维重建任务中常用的误差度量，用于评估相
// 机模型的准确性。
struct CamIntrCalib::ReprojErr
{
public:
    //定义构造函数
    ReprojErr(const cv::Point2f &observe_p_2d_, const cv::Point2f &world_p_3d_)
        : observe_p_2d(observe_p_2d_), world_p_3d(world_p_3d_) {}

    //这是一个模板函数，用于计算重投影误差。它重载了 operator()，使得 ReprojErr 实例可以像函数一样被调用。
    //operator() 用于定义代价函数的计算逻辑。代价函数是优化问题的核心，它定义了如何计算残差（即误差项）
    template <typename T>
    bool operator()(const T *const camera, const T *const K, const T *const dist_coeff, T *residual) const
    {
        //! camera传入的并不是6*1的矩阵，只是取前面6个元素
        //  camera[0,1,2] are the angle-axis rotation.
        //将三维世界点转换为齐次坐标p_3d
        T p_3d[3] = {static_cast<T>(world_p_3d.x), static_cast<T>(world_p_3d.y), static_cast<T>(0)};
        T p[3];
        //将p_3d旋转到相机坐标系下。
        ceres::AngleAxisRotatePoint(camera, p_3d, p);
        // camera[3,4,5] are the translation.
        //将平移向量 t 加到旋转后的点上，得到相机坐标系下的点 p。
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T x = p[0] / p[2];
        T y = p[1] / p[2];
        T r2 = x * x + y * y;

        //提取相机内参和畸变系数
        const T &alpha = K[0];
        const T &beta = K[1];
        const T &u0 = K[2];
        const T &v0 = K[3];

        const T &k1 = dist_coeff[0];
        const T &k2 = dist_coeff[1];

        //计算畸变后的归一化图像点坐标x_dist,y_dist
        T x_dist = x * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2);
        T y_dist = y * (static_cast<T>(1) + k1 * r2 + k2 * r2 * r2);

        //使用相机内参将畸变后的归一化图像点坐标转换为图像坐标u_dist,v_dist
        const T u_dist = alpha * x_dist + u0;
        const T v_dist = beta * y_dist + v0;

        //计算残差，即畸变后的图像点坐标与观测到的图像点坐标之间的差异。
        residual[0] = u_dist - static_cast<T>(observe_p_2d.x);
        residual[1] = v_dist - static_cast<T>(observe_p_2d.y);
        return true;
    }
    
    // 工厂函数，避免重复创建和析构实例
    //提供一个静态工厂函数 Create，用于创建 ceres::CostFunction 对象。
        //ceres::CostFunction: 是 Ceres Solver 中定义代价函数的基类。它是一个抽象类，
        // 提供了代价函数的基本接口。所有具体的代价函数类都需要继承自 ceres::CostFunction，
        // 并实现其纯虚函数。
        //ceres::AutoDiffCostFunction: 是 ceres::CostFunction 的一个具体实现，
        // 它利用模板和自动微分技术来自动计算代价函数的梯度
    static ceres::CostFunction *Create(const cv::Point2f &observe_p_2d_, const cv::Point2f &world_p_3d_)
    {
        //ceres::AutoDiffCostFunction 是一个模板类，用于自动计算代价函数的梯度。
        //它的构造函数需要一个 ReprojErr 的实例，这个实例包含了重投影误差的计算逻辑。  
        return new ceres::AutoDiffCostFunction<ReprojErr, 2, 6, 4, 2>(new ReprojErr(observe_p_2d_, world_p_3d_));
    }

private:
    const cv::Point2f observe_p_2d;
    const cv::Point2f world_p_3d;
};

//@func
//  使用 Ceres Solver 对相机参数进行优化。
void CamIntrCalib::Optimize()
{
    //创建一个 Ceres Solver 的 Problem 对象，用于存储优化问题的所有信息。
    ceres::Problem problem;
    //获取图像数量
    int pic_num = points_3d_vec_.size();
    //初始化相机内参参数
    double *K_para = new double[4];
    *(K_para + 0) = K_.at<double>(0, 0);
    *(K_para + 1) = K_.at<double>(1, 1);
    *(K_para + 2) = K_.at<double>(0, 2);
    *(K_para + 3) = K_.at<double>(1, 2);
    // 初始化畸变系数参数
    double *coeff_para = new double[2];
    *(coeff_para + 0) = dist_coef_.at<double>(0, 0);
    *(coeff_para + 1) = dist_coef_.at<double>(1, 0);
    //初始化相机外参参数
    double *cam_para = new double[6 * pic_num];
    for (int i = 0; i < pic_num; ++i)
    {
        const cv::Mat &R = R_vec_[i];
        const cv::Mat &t = t_vec_[i];
        cv::Mat angle_axis;
        //使用 cv::Rodrigues 函数将旋转矩阵 R 转换为角轴表示。
        cv::Rodrigues(R, angle_axis);

        *(cam_para + 6 * i + 0) = angle_axis.at<double>(0, 0);
        *(cam_para + 6 * i + 1) = angle_axis.at<double>(1, 0);
        *(cam_para + 6 * i + 2) = angle_axis.at<double>(2, 0);
        *(cam_para + 6 * i + 3) = t.at<double>(0, 0);
        *(cam_para + 6 * i + 4) = t.at<double>(1, 0);
        *(cam_para + 6 * i + 5) = t.at<double>(2, 0);
    }

    //遍历所有图像和图像中的每个点。
    for (int i = 0; i < pic_num; ++i)
    {
        const std::vector<cv::Point2f> &points_3ds = points_3d_vec_[i];
        const std::vector<cv::Point2f> &points_2ds = points_2d_vec_[i];
        for (size_t j = 0; j < points_3ds.size(); ++j)
        {
            double *cam_para_now = cam_para + 6 * i;
            //使用 ReprojErr::Create 工厂函数创建 ceres::CostFunction 对象。
            ceres::CostFunction *cost_function =
                ReprojErr::Create(points_2ds[j], points_3ds[j]);
            //将残差块添加到 Ceres Solver 的 Problem 对象中。
            problem.AddResidualBlock(cost_function, nullptr, cam_para_now, K_para, coeff_para);
        }
    }
    //配置 Ceres Solver 选项
    ceres::Solver::Options options;
    //启用cuda
    options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::CUDA;
    
    //options.linear_solver_type = ceres::DENSE_SCHUR;//选择线性求解器类型为 DENSE_SCHUR，适用于小规模问题。
    //options.minimizer_progress_to_stdout = true;    //将优化过程的进度输出到标准输出。
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;


    // 执行优化
    ceres::Solver::Summary summary;    //创建一个 ceres::Solver::Summary 对象，用于存储优化结果。
    ceres::Solve(options, &problem, &summary);//执行优化
    std::cout << "origin K:\n"
              << K_ << std::endl;
    std::cout << "origin dist_coeff\n:" << dist_coef_ << std::endl;


    K_.at<double>(0, 0) = *(K_para + 0);
    K_.at<double>(1, 1) = *(K_para + 1);
    K_.at<double>(0, 2) = *(K_para + 2);
    K_.at<double>(1, 2) = *(K_para + 3);
    dist_coef_.at<double>(0, 0) = *(coeff_para + 0);
    dist_coef_.at<double>(1, 0) = *(coeff_para + 1);
    std::cout << "ceres optimize K:\n"
              << K_ << std::endl;
    std::cout << "ceres optimize dist_coeff:\n"
              << dist_coef_ << std::endl;

    // update R_vec, t_vec
    for (size_t i = 0; i < R_vec_.size(); ++i)
    {
        double *T_para = cam_para + 6 * i;
        cv::Mat R_rid = cv::Mat::zeros(3, 1, CV_64F);
        R_rid.at<double>(0, 0) = *(T_para + 0);
        R_rid.at<double>(1, 0) = *(T_para + 1);
        R_rid.at<double>(2, 0) = *(T_para + 2);
        cv::Mat opt_R;
        cv::Rodrigues(R_rid, opt_R);
        R_vec_[i] = opt_R;
        t_vec_[i].at<double>(0, 0) = *(T_para + 3);
        t_vec_[i].at<double>(1, 0) = *(T_para + 4);
        t_vec_[i].at<double>(2, 0) = *(T_para + 5);
    }
}

//@func
//  将二维点转换为三维点（棋盘格的 z 坐标为 0）。
void CamIntrCalib::Transform2dTo3dPts(const std::vector<std::vector<cv::Point2f>> &points_3d_vec,
                                      std::vector<std::vector<cv::Point3f>> *points_3ds)
{
    points_3ds->clear();
    for (size_t i = 0; i < points_3d_vec_.size(); ++i)
    {
        std::vector<cv::Point3f> corner_3ds;
        for (const auto &p : points_3d_vec_[i])
        {
            cv::Point3f p_3d;
            p_3d.x = p.x;
            p_3d.y = p.y;
            p_3d.z = 0;
            corner_3ds.push_back(p_3d);
        }
        points_3ds->push_back(std::move(corner_3ds));
    }
}

//@func
//  计算两组二维点之间的平均误差。
double CamIntrCalib::CalcDiff(const std::vector<cv::Point2f> &points_2d_vec1,
                              const std::vector<cv::Point2f> &points_2d_vec2)
{
    double reproj_err = 0;
    int p_num = 0;
    for (size_t i = 0; i < points_2d_vec1.size(); ++i)
    {
        reproj_err += fabs(points_2d_vec1[i].x - points_2d_vec2[i].x);
        reproj_err += fabs(points_2d_vec1[i].y - points_2d_vec2[i].y);
        p_num++;
    }
    reproj_err /= static_cast<double>(2 * p_num);
    return reproj_err;
}

//@func
//  将三维点投影到二维图像平面，考虑畸变。
cv::Point2f CamIntrCalib::ReprojectPoint(const cv::Point3f &p_3d, const cv::Mat &R, const cv::Mat &t,
                                         const cv::Mat &K, const double k1, const double k2)
{
    cv::Mat p = cv::Mat(3, 1, CV_64F);
    p.at<double>(0, 0) = p_3d.x;
    p.at<double>(1, 0) = p_3d.y;
    p.at<double>(2, 0) = p_3d.z;
    cv::Mat trans_p = R * p + t;
    double x = trans_p.at<double>(0, 0) / trans_p.at<double>(2, 0);
    double y = trans_p.at<double>(1, 0) / trans_p.at<double>(2, 0);

    double r2 = x * x + y * y;

    const double alpha = K.at<double>(0, 0);
    const double beta = K.at<double>(1, 1);
    const double u0 = K.at<double>(0, 2);
    const double v0 = K.at<double>(1, 2);

    double x_dist = x * (1 + k1 * r2 + k2 * r2 * r2);
    double y_dist = y * (1 + k1 * r2 + k2 * r2 * r2);

    const double u_dist = alpha * x_dist + u0;
    const double v_dist = beta * y_dist + v0;
    cv::Point2f p_dist;
    p_dist.x = u_dist;
    p_dist.y = v_dist;
    return p_dist;
}

//@func
//  计算重投影误差。
double CamIntrCalib::CalcRepjErr()
{
    double reproj_err = 0;
    int p_num = 0;
    for (size_t i = 0; i < points_3d_vec_.size(); ++i)
    {
        for (size_t j = 0; j < points_3d_vec_[i].size(); ++j)
        {
            cv::Point3f p;
            p.x = points_3d_vec_[i][j].x;
            p.y = points_3d_vec_[i][j].y;
            p.z = 0;
            cv::Point2f reproj_p = ReprojectPoint(p, R_vec_[i], t_vec_[i],
                                                  K_, dist_coef_.at<double>(0, 0), dist_coef_.at<double>(1, 0));
            const cv::Point2f origin_p = points_2d_vec_[i][j];
            reproj_err += sqrt((origin_p.x - reproj_p.x) * (origin_p.x - reproj_p.x) + (origin_p.y - reproj_p.y) * (origin_p.y - reproj_p.y));
            ++p_num;
        }
    }
    reproj_err /= static_cast<double>(p_num);
    std::cout << "reprojection error: " << reproj_err << std::endl;
    return reproj_err;
}