
#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"
#include "kdtree.h"
#include "xform.h"

#include <highgui.h>
#include <io.h>
#include <iostream>
#include <opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;
#define OPTIONS ":o:m:i:s:c:r:n:b:dxh"


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

/******************************** Globals ************************************/

char* pname;
char* img_file_name;
char* out_file_name = NULL;
char* out_img_name = NULL;
int intvls = SIFT_INTVLS;
double sigma = SIFT_SIGMA;
double contr_thr = SIFT_CONTR_THR;
int curv_thr = SIFT_CURV_THR;
int img_dbl = SIFT_IMG_DBL;
int descr_width = SIFT_DESCR_WIDTH;
int descr_hist_bins = SIFT_DESCR_HIST_BINS;
int display = 1;

int main()
{
	clock_t start, finish,tmp_s,tmp_f;
 	IplImage* img1;
	struct feature* feat1;
	int n1 = 0;
	img1 = cvLoadImage("data//desk1.jpg", 1);
	start = clock();
	tmp_s = clock();
	n1 =  _sift_features(img1, &feat1, intvls, sigma, contr_thr, curv_thr,
		img_dbl, descr_width, descr_hist_bins);
	tmp_f = clock();
	cout << "total features: " << n1 << " for picture1 " << endl;
	cout << "spent " << tmp_f - tmp_s << "ms for picture1 to detect features!" << endl;
	//draw_features(img1, feat1, n1);
	//cvSaveImage("data//result//out1.jpg", img1);
	tmp_s = clock();
	IplImage* img2;
	struct feature* feat2;
	int n2 = 0;
	img2 = cvLoadImage("data//desk2.jpg", 1);
	n2 = _sift_features(img2, &feat2, intvls, sigma, contr_thr, curv_thr,
		img_dbl, descr_width, descr_hist_bins);
	tmp_f = clock();
	cout << "total features: " << n2 <<" for picture2 "<< endl;
	cout << "spent " << tmp_f - tmp_s << "ms for picture2 to detect features!" << endl;
	//draw_features(img2, feat2, n2);
	//cvSaveImage("data//result//out2.jpg", img2);

	//////////////matches//////////////////
	tmp_s = clock();
	IplImage* stacked;
	struct feature* feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int k, i, m = 0;

	stacked = stack_imgs(img1, img2);
	kd_root = kdtree_build(feat2, n2);
	for (i = 0; i < n1; i++)
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
		if (k == 2)
		{
			d0 = descr_dist_sq(feat, nbrs[0]);
			d1 = descr_dist_sq(feat, nbrs[1]);
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR)
			{
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
				pt2.y += img1->height;
				cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free(nbrs);
	}
	tmp_f = clock();
	finish = clock();
	cout << "total matches: " << m << endl;
	cout << "spent " << tmp_f - tmp_s << " ms for these 2 pics to match" << endl;
	cout << "total time : " << finish - start << endl;
	cvSaveImage("data//result//deskout.jpg", stacked);
	system("pause");

	CvMat* H;
	IplImage* xformed;
	//H = ransac_xform(feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
	//	homog_xfer_err, 3.0, NULL, NULL);
	struct feature** inliers;
	int n_inliers=0;
	H = ransac_xform(feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
	if (H)
	{
		//xformed = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
		//cvWarpPerspective(img1, xformed, H,
		//	CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
		//	cvScalarAll(0));
		//cvSaveImage("data//result//Xformed.jpg", xformed);
		//cvReleaseImage(&xformed);
		//cvReleaseMat(&H);
		cout << "经RANSAC算法筛选后的匹配点对个数：" << m << endl; //输出筛选后的匹配点对个数  

		int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图  

						  //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线  
		for (int i = 0; i<n_inliers; i++)
		{
			feat = inliers[i];//第i个特征点  
			pt2 = Point(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标  
			pt1 = Point(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)  
																				  //qDebug()<<"pt2:("<<pt2.x<<","<<pt2.y<<")--->pt1:("<<pt1.x<<","<<pt1.y<<")";//输出对应点对  

																				  //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系  
			if (pt2.x > pt1.x)
				invertNum++;

			pt2.y += img1->height;
			cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//在匹配图上画出连线  
		}

		cvSaveImage("ransac.jpg", stacked);
	}
	
	cvReleaseImage(&stacked);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	kdtree_release(kd_root);
	free(feat1);
	free(feat2);
	return 0;
}
