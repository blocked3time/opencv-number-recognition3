//2001223 정용규
//https://www.youtube.com/watch?v=JnqAQjuFL-s
#include <iostream>
#include<vector>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

int number_recognition(Mat img) {
    Mat lable, stats, cen, dst;
    double px = 0, py = 0;
    vector<vector<Point>> vvp;
    vector<vector<Point>> vvp2;
    vector<Vec4i> hierarchy;
    cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
    threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
    morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
    connectedComponentsWithStats(dst, lable, stats, cen);
    findContours(dst, vvp, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    if (vvp.size() == 0)
        return -1;
    double cgx = (cen.at<double>(1, 0) - stats.at<int>(1, 0)) / stats.at<int>(1, 2),
        cgy = (cen.at<double>(1, 1) - stats.at<int>(1, 1)) / stats.at<int>(1, 3);//center of gravity x,y
    if (vvp.size() == 3) {//외각선이 3일 떄 
        return 8;
    }
    else if (vvp.size() == 2) {
        for (int j = 0; j < vvp[1].size(); j++) {//2번째 외각선의 무게중심 구하기
            px += vvp[1][j].x;
            py += vvp[1][j].y;
        }
        px = ((px / vvp[1].size()) - stats.at<int>(1, 0)) / stats.at<int>(1, 2);
        py = ((py / vvp[1].size()) - stats.at<int>(1, 1)) / stats.at<int>(1, 3);

        if ((cgx >= 0.55 && cgy <= 0.53) && (py <= 0.42))//9
            return 9;
        else if ((cgy >= 0.5 || cgx >= 0.5) && py >= 0.6)//6
            return 6;
        else if ((cgx <= 0.47 && cgy <= 0.58) && (py <= 0.42 && px <= 0.42))//4
            return 4;
        else if (((0.4 <= cgx && cgx <= 0.6) && (0.4 <= cgy && cgy <= 0.6)) && ((0.4 <= px && px <= 0.6) && (0.4 <= py && py <= 0.6)))//무게 중심이 중간일 떄
            return 0;
        else return -1;
    }
    else if (vvp.size() == 1) {//외각선이 1개 일 떄
        int rt = 0, lt = 0, rb = 0, lb = 0;//우상좌상우하좌하 카운트
        double mx = cen.at<double>(1, 0);
        double  my = stats.at<int>(1, 1) + stats.at<int>(1, 3) / 2;
        line(dst, Point(mx, stats.at<int>(1, 1)),
            Point(mx, stats.at<int>(1, 1) + stats.at<int>(1, 3)), 255, 3);
        findContours(dst, vvp2, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
        if (vvp2.size() <= 2) {
            //세로로 선을 그었을 떄 다른 외각선 2개 이상 검출되지 않을시
            vvp2.clear();
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);//라인 없는 깨끗한 값 받아오기
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            line(dst, Point(stats.at<int>(1, 0), stats.at<int>(1, 1)),
                Point(stats.at<int>(1, 0) + stats.at<int>(1, 2), stats.at<int>(1, 1) + stats.at<int>(1, 3)), 255, 3);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 3);
            findContours(dst, vvp2, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            px = 0;
            py = 0;
            for (int i = 1; i < vvp2.size(); i++) {
                px = 0;
                py = 0;
                for (int j = 0; j < vvp2[i].size(); j++) {
                    px += vvp2[i][j].x;
                    py += vvp2[i][j].y;
                }
                px /= vvp2[i].size();
                py /= vvp2[i].size();
                dst.at<uchar>(py, px) = 255;//점 표시
                if (px < mx && py < my) lt++;
                else if (px > mx && py < my) rt++;
                else if (px < mx && py > my) lb++;
                else if (px > mx && py > my) rb++;
            }


            if (((vvp2.size() == 2 && (cgx >= 0.5 && cgy <= 0.4)) && (double)stats.at<int>(1, 3) / (double)stats.at<int>(1, 2) > 1) && (rb == 0 && lb == 0))//7
                return 7;
            else if ((vvp2.size() == 1) && (double)stats.at<int>(1, 3) / (double)stats.at<int>(1, 2) > 1.3)//1
                return 1;
            else if (vvp2.size() >= 2 &&(lb >= 1|| lt >= 1) && (cgx <= 0.5 && cgy <= 0.58))//4
                return 4;
            else if (lt >= 1 && rb >= 1) // 2 밑부분 겹쳐쓸 떄
                return 2;
            else
                return -1;
        }


        for (int i = 1; i < vvp2.size(); i++) {
            px = 0;
            py = 0;
            for (int j = 0; j < vvp2[i].size(); j++) {
                px += vvp2[i][j].x;
                py += vvp2[i][j].y;
            }
            px /= vvp2[i].size();
            py /= vvp2[i].size();
            dst.at<uchar>(py, px) = 255;//점 표시
            if (px < mx && py < my) lt++;
            else if (px > mx && py < my) rt++;
            else if (px < mx && py > my) lb++;
            else if (px > mx && py > my) rb++;
        }

        if (((lt >= 1 && rb >= 1) && (rt == 0 && lb == 0)) && cgy <= 0.55)//5
            return 5;
        else if ((rt >= 1 && lb >= 1) && (double)stats.at<int>(1, 3) / (double)stats.at<int>(1, 2) < 1.7)//2
            return 2;
        else  if ((rt >= 1 && rb >= 1) && (cgx >= 0.48))//3
            return 3;
        else  if ((cgx <= 0.5 && cgy <= 0.6))//4
            return 4;
        else return -1;
    }
    else return -1;
}
void mousecallback(int e, int x, int y, int f, void* u) {
    Mat img = *(Mat*)u, lable, stats, cen, dst;
    static Point op;
    vector<vector<Point>> vvp;
    vector<vector<Point>> vvp2;
    vector<Vec4i> hierarchy;
    Rect r(Point(0, 0), Point(499, 499));//영역 분할
    Rect s(Point(499, 0), Point(699, 99));
    Rect l(Point(499, 99), Point(699, 199));
    Rect c(Point(499, 199), Point(699, 299));
    Rect run(Point(499, 299), Point(699, 399));
    Rect ex(Point(399, 399), Point(699, 499));
    Rect f1(Point(699, 0), Point(899, 99));
    Rect f2(Point(699, 99), Point(899, 199));
    Rect f3(Point(699, 199), Point(899, 299));
    Rect f4(Point(699, 299), Point(899, 399));
    Rect f5(Point(699, 399), Point(899, 499));
    switch (e) {
    case EVENT_LBUTTONDOWN:
        if (r.contains(Point(x, y))) {
            op = Point(x, y);
        }
        else if (c.contains(Point(x, y))) {//클리어
            cout << "clear" << endl;
            img(Rect(Point(2, 2), Point(498, 498))).setTo(255);
        }
        else if (s.contains(Point(x, y))) {
            String s;
            cout << "파일 명 입력 :";
            cin >> s;
            resize(img(Rect(Point(2, 2), Point(498, 498))), dst, Size(500, 500), 0, 0);//리사이즈로 500x500사이즈 맞춤
            imwrite(s, dst);
            cout << s << " 저장됨" << endl;
        }
        else if (l.contains(Point(x, y))) {
            String s;
            cout << "파일 명 입력 :";
            cin >> s;
            dst = imread(s);
            if (s.empty()) {
                cerr << "이미지 불러오기 실패!" << endl;
                return;
            }
            cout << "iamge 불러옴" << endl;
            resize(dst, dst, Size(496, 496));
            dst.copyTo(img(Rect(Point(2, 2), Point(498, 498))));//사이즈 맞춤
        }
        else if (run.contains(Point(x, y))) {

            int num = number_recognition(img);
            if (num == -1) {
                cerr << "인식 불가" << endl;
                return;
            }
            else
                cout << "인식한 값" << num << endl;
            break;
        }
        else if (ex.contains(Point(x, y))) {
            cout << "종료";
            exit(1);
        }
        else if (f1.contains(Point(x, y))) {//비율
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            connectedComponentsWithStats(dst, lable, stats, cen);
            cout << "비율 (height)/(width)" << ((double)stats.at<int>(1, 3) / (double)stats.at<int>(1, 2)) << endl;
            break;
        }
        else if (f2.contains(Point(x, y))) {//외각선
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            findContours(dst, vvp, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            cout << "외각선의 개수" << vvp.size() << endl;
            break;

        }
        else if (f3.contains(Point(x, y))) {//무게중심/w or h
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            connectedComponentsWithStats(dst, lable, stats, cen);
            double cgx = cen.at<double>(1, 0) - stats.at<int>(1, 0), cgy = cen.at<double>(1, 1) - stats.at<int>(1, 1);//center of gravity x,y
            cout << "무게중심 : " << cgx << ':' << cgy << endl;
            cout << "무게 중심 좌표 x/w : y/h" << cgx / stats.at<int>(1, 2) << ':' << cgy / stats.at<int>(1, 3) << endl;
            break;
        }
        else if (f4.contains(Point(x, y))) {//세로 선으로 어서 검출된 외각선의 좌표를 출력
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            connectedComponentsWithStats(dst, lable, stats, cen);
            double mx = cen.at<double>(1, 0);
            double my = stats.at<int>(1, 1) + stats.at<int>(1, 3) / 2;
            line(dst, Point(mx, stats.at<int>(1, 1)),
                Point(mx, stats.at<int>(1, 1) + stats.at<int>(1, 3)), 255, 3);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            findContours(dst, vvp2, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            cout << "세로 중앙선을 그었을 때 외각선의 개수" << vvp2.size() << endl;
            int rt = 0, lt = 0, rb = 0, lb = 0;//우상좌상우하좌하 카운트
            if (vvp2.size() <= 2) {
                vvp2.clear();
                cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);//라인 없는 깨끗한 값 받아오기
                threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
                morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
                line(dst, Point(stats.at<int>(1, 0), stats.at<int>(1, 1)),
                    Point(stats.at<int>(1, 0) + stats.at<int>(1, 2), stats.at<int>(1, 1) + stats.at<int>(1, 3)), 255, 3);
                morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 5);
                findContours(dst, vvp2, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
                double px = 0;
                double py = 0;
                for (int i = 1; i < vvp2.size(); i++) {
                    px = 0;
                    py = 0;
                    for (int j = 0; j < vvp2[i].size(); j++) {
                        px += vvp2[i][j].x;
                        py += vvp2[i][j].y;
                    }
                    px /= vvp2[i].size();
                    py /= vvp2[i].size();
                    dst.at<uchar>(py, px) = 255;//점 표시

                    if (px > mx && py < my) rt++;
                    else if (px < mx && py < my) lt++;
                    else if (px < mx && py > my) lb++;
                    else if (px > mx && py > my) rb++;
                }
                imshow("dst", dst);
                cout << "대각선으로 그었을 떄 외각선의 개수 : " << vvp2.size() << endl;;
                cout << "lt rt\nld rd :\n " << lt << rt << endl << lb << rb << endl;
                break;
            }
            //lt rt
            //ld rd
            for (int i = 1; i < vvp2.size(); i++) {
                double px = 0, py = 0;
                for (int j = 0; j < vvp2[i].size(); j++) {
                    px += vvp2[i][j].x;
                    py += vvp2[i][j].y;
                }
                px /= vvp2[i].size();
                py /= vvp2[i].size();
                dst.at<uchar>(py, px) = 255;//점 표시
                if (px < mx && py < my) lt++;
                else if (px > mx && py < my) rt++;
                else if (px < mx && py > my) lb++;
                else if (px > mx && py > my) rb++;
            }
            cout << "lt rt\nld rd :\n " << lt << rt << endl << lb << rb << endl;
            imshow("dst", dst);
            break;
        }
        else if (f5.contains(Point(x, y))) {
            double px = 0, py = 0;
            cvtColor(img(Rect(Point(2, 2), Point(498, 498))), dst, COLOR_BGR2GRAY);
            threshold(dst, dst, 128, 255, THRESH_BINARY_INV);
            morphologyEx(dst, dst, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
            connectedComponentsWithStats(dst, lable, stats, cen);
            findContours(dst, vvp, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            if (vvp.size() < 2) {
                cout << "2번쨰 외각선 없음 " << endl;
                return;
            }
            for (int j = 0; j < vvp[1].size(); j++) {//2번째 외각선의 무게중심 구하기
                px += vvp[1][j].x;
                py += vvp[1][j].y;
            }
            px = (px / vvp[1].size() - stats.at<int>(1, 0)) / stats.at<int>(1, 2);
            py = (py / vvp[1].size() - stats.at<int>(1, 1)) / stats.at<int>(1, 3);
            cout << "2번째 외각선의 무게중심 x/w:y/h" << px << ':' << py << endl;
            break;
        }
    case EVENT_MOUSEMOVE:
        if (f & EVENT_FLAG_LBUTTON && r.contains(Point(x, y))) {
            line(img, op, Point(x, y), Scalar(0, 0, 0), 3);
            op = Point(x, y);
        }
        break;
    default:
        break;
    }
    imshow("img", img);
}
int main(void) {
    Mat img(500, 900, CV_8UC3, Scalar(255, 255, 255));
    string s[] = { "Save","Load","Clear","Run","Exit" };
    line(img, Point(499, 0), Point(499, 499), 0, 2);
    line(img, Point(699, 0), Point(699, 499), 0, 2);
    rectangle(img, Rect(Point(0, 0), Point(img.cols, img.rows)), 0, 2);
    for (int i = 0; i < 5; i++) {
        line(img, Point(499, i * 100), Point(899, i * 100), Scalar(0, 0, 0), 2);
        Size sizetext = getTextSize(s[i], FONT_HERSHEY_COMPLEX, 2, 3, 0);
        putText(img, s[i], Point(499 + (200 - sizetext.width) / 2, i * 100 + (100 + sizetext.height) / 2), FONT_HERSHEY_COMPLEX, 2, 0, 3);
        string sf = format("ft%d", i + 1);
        sizetext = getTextSize(sf, FONT_HERSHEY_COMPLEX, 2, 3, 0);
        putText(img, sf, Point(699 + (200 - sizetext.width) / 2, i * 100 + (100 + sizetext.height) / 2), FONT_HERSHEY_COMPLEX, 2, 0, 3);
    }
    imshow("img", img);
    setMouseCallback("img", mousecallback, &img);
    waitKey();
    return 0;
}