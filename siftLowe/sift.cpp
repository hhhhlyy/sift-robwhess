
#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"

#include <cxcore.h>
#include <cv.h>

/************************* Local Function Prototypes *************************/

static IplImage* create_init_img(IplImage*, int, double);
static IplImage* convert_to_gray32(IplImage*);
static IplImage*** build_gauss_pyr(IplImage*, int, int, double);
static IplImage* downsample(IplImage*);
static IplImage*** build_dog_pyr(IplImage***, int, int);
static CvSeq* scale_space_extrema(IplImage***, int, int, double, int,
	CvMemStorage*);
static int is_extremum(IplImage***, int, int, int, int);
static struct feature* interp_extremum(IplImage***, int, int, int, int, int,
	double);
static void interp_step(IplImage***, int, int, int, int, double*, double*,
	double*);
static CvMat* deriv_3D(IplImage***, int, int, int, int);
static CvMat* hessian_3D(IplImage***, int, int, int, int);
static double interp_contr(IplImage***, int, int, int, int, double, double,
	double);
static struct feature* new_feature(void);
static int is_too_edge_like(IplImage*, int, int, int);
static void calc_feature_scales(CvSeq*, double, int);
static void adjust_for_img_dbl(CvSeq*);
static void calc_feature_oris(CvSeq*, IplImage***);
static double* ori_hist(IplImage*, int, int, int, int, double);
static int calc_grad_mag_ori(IplImage*, int, int, double*, double*);
static void smooth_ori_hist(double*, int);
static double dominant_ori(double*, int);
static void add_good_ori_features(CvSeq*, double*, int, double,
	struct feature*);
static struct feature* clone_feature(struct feature*);
static void compute_descriptors(CvSeq*, IplImage***, int, int);
static double*** descr_hist(IplImage*, int, int, double, double, int, int);
static void interp_hist_entry(double***, double, double, double, double, int,
	int);
static void hist_to_descr(double***, int, int, struct feature*);
static void normalize_descr(struct feature*);
static int feature_cmp(void*, void*, void*);
static void release_descr_hist(double****, int);
static void release_pyr(IplImage****, int, int);
static float pixval32f(IplImage* img, int r, int c);


/*
Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )


int sift_features(IplImage* img, struct feature** feat)
{
	return _sift_features(img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
		SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
		SIFT_DESCR_HIST_BINS);
}

int _sift_features(IplImage* img, struct feature** feat, int intvls,
	double sigma, double contr_thr, int curv_thr,
	int img_dbl, int descr_width, int descr_hist_bins)
{
	IplImage* init_img;

	init_img = create_init_img(img, img_dbl, sigma);
	int octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;

	IplImage*** gauss_pyr, *** dog_pyr;
	dog_pyr = (IplImage ***)calloc(octvs, sizeof(IplImage**));
	CvMemStorage* storage;
	CvSeq* features;
	int  i, n = 0;

	/* build scale space pyramid; smallest dimension of top level is ~4 pixels */
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);
	storage = cvCreateMemStorage(0);
	features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr,
		curv_thr, storage);
	calc_feature_scales(features, sigma, intvls);
	if (img_dbl)
		adjust_for_img_dbl(features);
	calc_feature_oris(features, gauss_pyr);
	compute_descriptors(features, gauss_pyr, descr_width, descr_hist_bins);
	cvSeqSort(features, (CvCmpFunc)feature_cmp, NULL);
	n = features->total;
	*feat = (feature*)calloc(n, sizeof(struct feature));
	*feat = (feature*)cvCvtSeqToArray(features, *feat, CV_WHOLE_SEQ);
	for (i = 0; i < n; i++)
	{
		free((*feat)[i].feature_data);
		(*feat)[i].feature_data = NULL;
	}
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&init_img);
	release_pyr(&gauss_pyr, octvs, intvls + 3);
	release_pyr(&dog_pyr, octvs, intvls + 2);
	return n;
}

static IplImage* create_init_img(IplImage* img, int img_dbl, double sigma)
{
	IplImage* gray, *dbl;
	double sig_diff;
	gray = convert_to_gray32(img);
	if (img_dbl)
	{
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
		dbl = cvCreateImage(cvSize(img->width * 2, img->height * 2),
			IPL_DEPTH_32F, 1);
		cvResize(gray, dbl, CV_INTER_CUBIC);
		cvSmooth(dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);
		cvReleaseImage(&gray);
		return dbl;
	}
	else
	{
		sig_diff = sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
		cvSmooth(gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff);
		return gray;
	}
}

static IplImage* convert_to_gray32(IplImage* img)
{
	IplImage* gray8, *gray32;

	gray32 = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
	//这里删掉了源代码中对原始就是灰度图的判断
	gray8 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	cvCvtColor(img, gray8, CV_BGR2GRAY);
	cvConvertScale(gray8, gray32, 1.0 / 255.0, 0);

	cvReleaseImage(&gray8);
	return gray32;
}

static IplImage*** build_gauss_pyr(IplImage* base, int octvs,
	int intvls, double sigma)
{
	IplImage*** gauss_pyr;
	const int _intvls = intvls;
	double *sig = new double[_intvls + 3];
	double sig_total, sig_prev, k;
	int i, o;

	gauss_pyr = (IplImage ***) calloc(octvs, sizeof(IplImage**));
	for (i = 0; i < octvs; i++)
		gauss_pyr[i] = (IplImage **) calloc(intvls + 3, sizeof(IplImage *));

	/*
	precompute Gaussian sigmas using the following formula:

	\sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2

	sig[i] is the incremental sigma value needed to compute
	the actual sigma of level i. Keeping track of incremental
	sigmas vs. total sigmas keeps the gaussian kernel small.
	*/
	k = pow(2.0, 1.0 / intvls);
	sig[0] = sigma;
	sig[1] = sigma * sqrt(k*k - 1);
	for (i = 2; i < intvls + 3; i++)
		sig[i] = sig[i - 1] * k;

	for (o = 0; o < octvs; o++)
		for (i = 0; i < intvls + 3; i++)
		{
			if (o == 0 && i == 0)
				gauss_pyr[o][i] = cvCloneImage(base);

			/* base of new octvave is halved image from end of previous octave */
			else if (i == 0)
				gauss_pyr[o][i] = downsample(gauss_pyr[o - 1][intvls]);

			/* blur the current octave's last image to create the next one */
			else
			{
				gauss_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i - 1]),
					IPL_DEPTH_32F, 1);
				cvSmooth(gauss_pyr[o][i - 1], gauss_pyr[o][i],
					CV_GAUSSIAN, 0, 0, sig[i], sig[i]);
			}
		}
	delete[] sig;
	return gauss_pyr;
}

static IplImage* downsample(IplImage* img)
{
	IplImage* smaller = cvCreateImage(cvSize(img->width / 2, img->height / 2),
		img->depth, img->nChannels);
	cvResize(img, smaller, CV_INTER_NN);

	return smaller;
}

static IplImage*** build_dog_pyr(IplImage*** gauss_pyr, int octvs, int intvls)
{
	IplImage*** dog_pyr;
	int i, o;

	dog_pyr = (IplImage ***)calloc(octvs, sizeof(IplImage**));
	for (i = 0; i < octvs; i++)
		dog_pyr[i] = (IplImage **)calloc(intvls + 2, sizeof(IplImage*));

	for (o = 0; o < octvs; o++)
		for (i = 0; i < intvls + 2; i++)
		{
			dog_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i]),
				IPL_DEPTH_32F, 1);
			cvSub(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i], NULL);
		}

	return dog_pyr;
}

static CvSeq* scale_space_extrema(IplImage*** dog_pyr, int octvs, int intvls,
	double contr_thr, int curv_thr,
	CvMemStorage* storage)
{
	CvSeq* features;
	double prelim_contr_thr = 0.5 * contr_thr / intvls;
	struct feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;
	unsigned long* feature_mat;

	features = cvCreateSeq(0, sizeof(CvSeq), sizeof(struct feature), storage);
	for (o = 0; o < octvs; o++)
	{
		feature_mat =(unsigned long*)calloc(dog_pyr[o][0]->height * dog_pyr[o][0]->width, sizeof(unsigned long));
		for (i = 1; i <= intvls; i++)
			for (r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height - SIFT_IMG_BORDER; r++)
				for (c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width - SIFT_IMG_BORDER; c++)
					if (abs(pixval32f(dog_pyr[o][i], r, c)) > prelim_contr_thr)//ABS->abs 
						if (is_extremum(dog_pyr, o, i, r, c))
						{
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
							if (feat)
							{
								ddata = feat_detection_data(feat);
								if (!is_too_edge_like(dog_pyr[ddata->octv][ddata->intvl],
									ddata->r, ddata->c, curv_thr))
								{
									if (ddata->intvl > sizeof(unsigned long))
										cvSeqPush(features, feat);
									else if ((feature_mat[dog_pyr[o][0]->width * ddata->r + ddata->c] & (1 << ddata->intvl - 1)) == 0)
									{
										cvSeqPush(features, feat);
										feature_mat[dog_pyr[o][0]->width * ddata->r + ddata->c] += 1 << ddata->intvl - 1;
									}
								}
								else
									free(ddata);
								free(feat);
							}
						}
		free(feature_mat);
	}
	return features;
}

//static float pixval32f(IplImage* img, int r, int c)
//{
//	return ((float*)(img->imageData + img->widthStep*r))[c];
//}

static void calc_feature_scales(CvSeq* features, double sigma, int intvls)
{
	struct feature* feat;
	struct detection_data* ddata;
	double intvl;
	int i, n;

	n = features->total;
	for (i = 0; i < n; i++)
	{
		feat = CV_GET_SEQ_ELEM(struct feature, features, i);
		ddata = feat_detection_data(feat);
		intvl = ddata->intvl + ddata->subintvl;
		feat->scl = sigma * pow(2.0, ddata->octv + intvl / intvls);
		ddata->scl_octv = sigma * pow(2.0, intvl / intvls);
	}
}

static void adjust_for_img_dbl(CvSeq* features)
{
	struct feature* feat;
	int i, n;

	n = features->total;
	for (i = 0; i < n; i++)
	{
		feat = CV_GET_SEQ_ELEM(struct feature, features, i);
		feat->x /= 2.0;
		feat->y /= 2.0;
		feat->scl /= 2.0;
		feat->img_pt.x /= 2.0;
		feat->img_pt.y /= 2.0;
	}
}

static void calc_feature_oris(CvSeq* features, IplImage*** gauss_pyr)
{
	struct feature* feat;
	struct detection_data* ddata;
	double* hist;
	double omax;
	int i, j, n = features->total;

	for (i = 0; i < n; i++)
	{
		feat = (feature*)malloc(sizeof(struct feature));
		cvSeqPopFront(features, feat);
		ddata = feat_detection_data(feat);
		hist = ori_hist(gauss_pyr[ddata->octv][ddata->intvl],
			ddata->r, ddata->c, SIFT_ORI_HIST_BINS,
			cvRound(SIFT_ORI_RADIUS * ddata->scl_octv),
			SIFT_ORI_SIG_FCTR * ddata->scl_octv);
		for (j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
			smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);
		omax = dominant_ori(hist, SIFT_ORI_HIST_BINS);
		add_good_ori_features(features, hist, SIFT_ORI_HIST_BINS,
			omax * SIFT_ORI_PEAK_RATIO, feat);
		free(ddata);
		free(feat);
		free(hist);
	}
}

static void compute_descriptors(CvSeq* features, IplImage*** gauss_pyr, int d,
	int n)
{
	struct feature* feat;
	struct detection_data* ddata;
	double*** hist;
	int i, k = features->total;

	for (i = 0; i < k; i++)
	{
		feat = CV_GET_SEQ_ELEM(struct feature, features, i);
		ddata = feat_detection_data(feat);
		hist = descr_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
			ddata->c, feat->ori, ddata->scl_octv, d, n);
		hist_to_descr(hist, d, n, feat);
		release_descr_hist(&hist, d);
	}
}

static int is_extremum(IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	double val = pixval32f(dog_pyr[octv][intvl], r, c);
	int i, j, k;

	/* check for maximum */
	if (val > 0)
	{
		for (i = -1; i <= 1; i++)
			for (j = -1; j <= 1; j++)
				for (k = -1; k <= 1; k++)
					if (val < pixval32f(dog_pyr[octv][intvl + i], r + j, c + k))
						return 0;
	}

	/* check for minimum */
	else
	{
		for (i = -1; i <= 1; i++)
			for (j = -1; j <= 1; j++)
				for (k = -1; k <= 1; k++)
					if (val > pixval32f(dog_pyr[octv][intvl + i], r + j, c + k))
						return 0;
	}

	return 1;
}

static struct feature* interp_extremum(IplImage*** dog_pyr, int octv,
	int intvl, int r, int c, int intvls,
	double contr_thr)
{
	struct feature* feat;
	struct detection_data* ddata;
	double xi, xr, xc, contr;
	int i = 0;

	while (i < SIFT_MAX_INTERP_STEPS)
	{
		interp_step(dog_pyr, octv, intvl, r, c, &xi, &xr, &xc);
		if (abs(xi) < 0.5  &&  abs(xr) < 0.5  &&  abs(xc) < 0.5)
			break;

		c += cvRound(xc);
		r += cvRound(xr);
		intvl += cvRound(xi);

		if (intvl < 1 ||
			intvl > intvls ||
			c < SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER ||
			c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER ||
			r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER)
		{
			return NULL;
		}

		i++;
	}

	/* ensure convergence of interpolation */
	if (i >= SIFT_MAX_INTERP_STEPS)
		return NULL;

	contr = interp_contr(dog_pyr, octv, intvl, r, c, xi, xr, xc);
	if (abs(contr) < contr_thr / intvls)
		return NULL;

	feat = new_feature();
	ddata = feat_detection_data(feat);
	feat->img_pt.x = feat->x = (c + xc) * pow(2.0, octv);
	feat->img_pt.y = feat->y = (r + xr) * pow(2.0, octv);
	ddata->r = r;
	ddata->c = c;
	ddata->octv = octv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;

	return feat;
}

static void interp_step(IplImage*** dog_pyr, int octv, int intvl, int r, int c,
	double* xi, double* xr, double* xc)
{
	CvMat* dD, *H, *H_inv, X;
	double x[3] = { 0 };

	dD = deriv_3D(dog_pyr, octv, intvl, r, c);
	H = hessian_3D(dog_pyr, octv, intvl, r, c);
	H_inv = cvCreateMat(3, 3, CV_64FC1);
	cvInvert(H, H_inv, CV_SVD);
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	cvGEMM(H_inv, dD, -1, NULL, 0, &X, 0);

	cvReleaseMat(&dD);
	cvReleaseMat(&H);
	cvReleaseMat(&H_inv);

	*xi = x[2];
	*xr = x[1];
	*xc = x[0];
}

static CvMat* deriv_3D(IplImage*** dog_pyr, int octv, int intvl, int r, int c)
{
	CvMat* dI;
	double dx, dy, ds;

	dx = (pixval32f(dog_pyr[octv][intvl], r, c + 1) -
		pixval32f(dog_pyr[octv][intvl], r, c - 1)) / 2.0;
	dy = (pixval32f(dog_pyr[octv][intvl], r + 1, c) -
		pixval32f(dog_pyr[octv][intvl], r - 1, c)) / 2.0;
	ds = (pixval32f(dog_pyr[octv][intvl + 1], r, c) -
		pixval32f(dog_pyr[octv][intvl - 1], r, c)) / 2.0;

	dI = cvCreateMat(3, 1, CV_64FC1);
	cvmSet(dI, 0, 0, dx);
	cvmSet(dI, 1, 0, dy);
	cvmSet(dI, 2, 0, ds);

	return dI;
}

static CvMat* hessian_3D(IplImage*** dog_pyr, int octv, int intvl, int r,
	int c)
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f(dog_pyr[octv][intvl], r, c);
	dxx = (pixval32f(dog_pyr[octv][intvl], r, c + 1) +
		pixval32f(dog_pyr[octv][intvl], r, c - 1) - 2 * v);
	dyy = (pixval32f(dog_pyr[octv][intvl], r + 1, c) +
		pixval32f(dog_pyr[octv][intvl], r - 1, c) - 2 * v);
	dss = (pixval32f(dog_pyr[octv][intvl + 1], r, c) +
		pixval32f(dog_pyr[octv][intvl - 1], r, c) - 2 * v);
	dxy = (pixval32f(dog_pyr[octv][intvl], r + 1, c + 1) -
		pixval32f(dog_pyr[octv][intvl], r + 1, c - 1) -
		pixval32f(dog_pyr[octv][intvl], r - 1, c + 1) +
		pixval32f(dog_pyr[octv][intvl], r - 1, c - 1)) / 4.0;
	dxs = (pixval32f(dog_pyr[octv][intvl + 1], r, c + 1) -
		pixval32f(dog_pyr[octv][intvl + 1], r, c - 1) -
		pixval32f(dog_pyr[octv][intvl - 1], r, c + 1) +
		pixval32f(dog_pyr[octv][intvl - 1], r, c - 1)) / 4.0;
	dys = (pixval32f(dog_pyr[octv][intvl + 1], r + 1, c) -
		pixval32f(dog_pyr[octv][intvl + 1], r - 1, c) -
		pixval32f(dog_pyr[octv][intvl - 1], r + 1, c) +
		pixval32f(dog_pyr[octv][intvl - 1], r - 1, c)) / 4.0;

	H = cvCreateMat(3, 3, CV_64FC1);
	cvmSet(H, 0, 0, dxx);
	cvmSet(H, 0, 1, dxy);
	cvmSet(H, 0, 2, dxs);
	cvmSet(H, 1, 0, dxy);
	cvmSet(H, 1, 1, dyy);
	cvmSet(H, 1, 2, dys);
	cvmSet(H, 2, 0, dxs);
	cvmSet(H, 2, 1, dys);
	cvmSet(H, 2, 2, dss);

	return H;
}

static double interp_contr(IplImage*** dog_pyr, int octv, int intvl, int r,
	int c, double xi, double xr, double xc)
{
	CvMat* dD, X, T;
	double t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	cvInitMatHeader(&T, 1, 1, CV_64FC1, t, CV_AUTOSTEP);
	dD = deriv_3D(dog_pyr, octv, intvl, r, c);
	cvGEMM(dD, &X, 1, NULL, 0, &T, CV_GEMM_A_T);
	cvReleaseMat(&dD);

	return pixval32f(dog_pyr[octv][intvl], r, c) + t[0] * 0.5;
}

static struct feature* new_feature(void)
{
	struct feature* feat;
	struct detection_data* ddata;

	feat = (feature*)malloc(sizeof(struct feature));
	memset(feat, 0, sizeof(struct feature));
	ddata = (detection_data*)malloc(sizeof(struct detection_data));
	memset(ddata, 0, sizeof(struct detection_data));
	feat->feature_data = ddata;
	feat->type = FEATURE_LOWE;

	return feat;
}

static int is_too_edge_like(IplImage* dog_img, int r, int c, int curv_thr)
{
	double d, dxx, dyy, dxy, tr, det;

	/* principal curvatures are computed using the trace and det of Hessian */
	d = pixval32f(dog_img, r, c);
	dxx = pixval32f(dog_img, r, c + 1) + pixval32f(dog_img, r, c - 1) - 2 * d;
	dyy = pixval32f(dog_img, r + 1, c) + pixval32f(dog_img, r - 1, c) - 2 * d;
	dxy = (pixval32f(dog_img, r + 1, c + 1) - pixval32f(dog_img, r + 1, c - 1) -
		pixval32f(dog_img, r - 1, c + 1) + pixval32f(dog_img, r - 1, c - 1)) / 4.0;
	tr = dxx + dyy;
	det = dxx * dyy - dxy * dxy;

	/* negative determinant -> curvatures have different signs; reject feature */
	if (det <= 0)
		return 1;

	if (tr * tr / det < (curv_thr + 1.0)*(curv_thr + 1.0) / curv_thr)
		return 0;
	return 1;
}

static double* ori_hist(IplImage* img, int r, int c, int n, int rad,
	double sigma)
{
	double* hist;
	double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	hist = (double*)calloc(n, sizeof(double));
	exp_denom = 2.0 * sigma * sigma;
	for (i = -rad; i <= rad; i++)
		for (j = -rad; j <= rad; j++)
			if (calc_grad_mag_ori(img, r + i, c + j, &mag, &ori))
			{
				w = exp(-(i*i + j*j) / exp_denom);
				bin = cvRound(n * (ori + CV_PI) / PI2);
				bin = (bin < n) ? bin : 0;
				hist[bin] += w * mag;
			}

	return hist;
}

static int calc_grad_mag_ori(IplImage* img, int r, int c, double* mag,
	double* ori)
{
	double dx, dy;

	if (r > 0 && r < img->height - 1 && c > 0 && c < img->width - 1)
	{
		dx = pixval32f(img, r, c + 1) - pixval32f(img, r, c - 1);
		dy = pixval32f(img, r - 1, c) - pixval32f(img, r + 1, c);
		*mag = sqrt(dx*dx + dy*dy);
		*ori = atan2(dy, dx);
		return 1;
	}

	else
		return 0;
}

static void smooth_ori_hist(double* hist, int n)
{
	double prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n - 1];
	for (i = 0; i < n; i++)
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * ((i + 1 == n) ? h0 : hist[i + 1]);
		prev = tmp;
	}
}

static double dominant_ori(double* hist, int n)
{
	double omax;
	int maxbin, i;

	omax = hist[0];
	maxbin = 0;
	for (i = 1; i < n; i++)
		if (hist[i] > omax)
		{
			omax = hist[i];
			maxbin = i;
		}
	return omax;
}

static void add_good_ori_features(CvSeq* features, double* hist, int n,
	double mag_thr, struct feature* feat)
{
	struct feature* new_feat;
	double bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	for (i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + interp_hist_peak(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? n + bin : (bin >= n) ? bin - n : bin;
			new_feat = clone_feature(feat);
			new_feat->ori = ((PI2 * bin) / n) - CV_PI;
			cvSeqPush(features, new_feat);
			free(new_feat);
		}
	}
}

static struct feature* clone_feature(struct feature* feat)
{
	struct feature* new_feat;
	struct detection_data* ddata;

	new_feat = new_feature();
	ddata = feat_detection_data(new_feat);
	memcpy(new_feat, feat, sizeof(struct feature));
	memcpy(ddata, feat_detection_data(feat), sizeof(struct detection_data));
	new_feat->feature_data = ddata;

	return new_feat;
}

static double*** descr_hist(IplImage* img, int r, int c, double ori,
	double scl, int d, int n)
{
	double*** hist;
	double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
		grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
	int radius, i, j;

	hist = (double***)calloc(d, sizeof(double**));
	for (i = 0; i < d; i++)
	{
		hist[i] = (double**)calloc(d, sizeof(double*));
		for (j = 0; j < d; j++)
			hist[i][j] = (double*)calloc(n, sizeof(double));
	}

	cos_t = cos(ori);
	sin_t = sin(ori);
	bins_per_rad = n / PI2;
	exp_denom = d * d * 0.5;
	hist_width = SIFT_DESCR_SCL_FCTR * scl;
	radius = hist_width * sqrt(2) * (d + 1.0) * 0.5 + 0.5;
	for (i = -radius; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
			/*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
			*/
			c_rot = (j * cos_t - i * sin_t) / hist_width;
			r_rot = (j * sin_t + i * cos_t) / hist_width;
			rbin = r_rot + d / 2 - 0.5;
			cbin = c_rot + d / 2 - 0.5;

			if (rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d)
				if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori))
				{
					grad_ori -= ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					obin = grad_ori * bins_per_rad;
					w = exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
					interp_hist_entry(hist, rbin, cbin, obin, grad_mag * w, d, n);
				}
		}

	return hist;
}

static void interp_hist_entry(double*** hist, double rbin, double cbin,
	double obin, double mag, int d, int n)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, *h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(rbin);
	c0 = cvFloor(cbin);
	o0 = cvFloor(obin);
	d_r = rbin - r0;
	d_c = cbin - c0;
	d_o = obin - o0;

	/*
	The entry is distributed into up to 8 bins.  Each entry into a bin
	is multiplied by a weight of 1 - d for each dimension, where d is the
	distance from the center value of the bin measured in bin units.
	*/
	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % n;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}
}

static void hist_to_descr(double*** hist, int d, int n, struct feature* feat)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < d; r++)
		for (c = 0; c < d; c++)
			for (o = 0; o < n; o++)
				feat->descr[k++] = hist[r][c][o];

	feat->d = k;
	normalize_descr(feat);
	for (i = 0; i < k; i++)
		if (feat->descr[i] > SIFT_DESCR_MAG_THR)
			feat->descr[i] = SIFT_DESCR_MAG_THR;
	normalize_descr(feat);

	/* convert floating-point descriptor to integer valued descriptor */
	for (i = 0; i < k; i++)
	{
		int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
		feat->descr[i] = MIN(255, int_val);
	}
}

static void normalize_descr(struct feature* feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat->d;

	for (i = 0; i < d; i++)
	{
		cur = feat->descr[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat->descr[i] *= len_inv;
}

static int feature_cmp(void* feat1, void* feat2, void* param)
{
	struct feature* f1 = (struct feature*) feat1;
	struct feature* f2 = (struct feature*) feat2;

	if (f1->scl < f2->scl)
		return 1;
	if (f1->scl > f2->scl)
		return -1;
	return 0;
}

static void release_descr_hist(double**** hist, int d)
{
	int i, j;

	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			free((*hist)[i][j]);
		free((*hist)[i]);
	}
	free(*hist);
	*hist = NULL;
}

static void release_pyr(IplImage**** pyr, int octvs, int n)
{
	int i, j;
	for (i = 0; i < octvs; i++)
	{
		for (j = 0; j < n; j++)
			cvReleaseImage(&(*pyr)[i][j]);
		free((*pyr)[i]);
	}
	free(*pyr);
	*pyr = NULL;
}
