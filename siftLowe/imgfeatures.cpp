
#include "utils.h"
#include "imgfeatures.h"

#include <opencv.hpp>
using namespace cv;

static void draw_lowe_features(IplImage* img, struct feature* feat, int n);

static void draw_lowe_feature(IplImage* img, struct feature* feat,
	CvScalar color);

void draw_features(IplImage* img, struct feature* feat, int n)
{
	int type;

	if (n <= 0 || !feat)
	{
		fprintf(stderr, "Warning: no features to draw, %s line %d\n",
			__FILE__, __LINE__);
		return;
	}
	draw_lowe_features(img, feat, n);
}
static void draw_lowe_features(IplImage* img, struct feature* feat, int n)
{
	CvScalar color = CV_RGB(255, 255, 255);
	int i;

	if (img->nChannels > 1)
		color = FEATURE_LOWE_COLOR;
	for (i = 0; i < n; i++)
		draw_lowe_feature(img, feat + i, color);
}

static void draw_lowe_feature(IplImage* img, struct feature* feat,
	CvScalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound(feat->x);
	start_y = cvRound(feat->y);
	scl = feat->scl;
	ori = feat->ori;
	len = cvRound(scl * scale);
	hlen = cvRound(scl * hscale);
	blen = len - hlen;
	end_x = cvRound(len *  cos(ori)) + start_x;
	end_y = cvRound(len * -sin(ori)) + start_y;
	h1_x = cvRound(blen *  cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen * -sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen *  cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen * -sin(ori - CV_PI / 18.0)) + start_y;
	start = cvPoint(start_x, start_y);
	end = cvPoint(end_x, end_y);
	h1 = cvPoint(h1_x, h1_y);
	h2 = cvPoint(h2_x, h2_y);

	cvLine(img, start, end, color, 1, 8, 0);
	cvLine(img, end, h1, color, 1, 8, 0);
	cvLine(img, end, h2, color, 1, 8, 0);
}

double descr_dist_sq(struct feature* f1, struct feature* f2)
{
	double diff, dsq = 0;
	double* descr1, *descr2;
	int i, d;

	d = f1->d;
	if (f2->d != d)
		return DBL_MAX;
	descr1 = f1->descr;
	descr2 = f2->descr;

	for (i = 0; i < d; i++)
	{
		diff = descr1[i] - descr2[i];
		dsq += diff*diff;
	}
	return dsq;
}
