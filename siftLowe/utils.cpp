#include "utils.h"

#include <opencv.hpp>

/*
Combines two images by scacking one on top of the other

@param img1 top image
@param img2 bottom image

@return Returns the image resulting from stacking \a img1 on top if \a img2
*/
extern IplImage* stack_imgs(IplImage* img1, IplImage* img2)
{
	IplImage* stacked = cvCreateImage(cvSize(MAX(img1->width, img2->width),
		img1->height + img2->height),
		IPL_DEPTH_8U, 3);

	cvZero(stacked);
	cvSetImageROI(stacked, cvRect(0, 0, img1->width, img1->height));
	cvAdd(img1, stacked, stacked, NULL);
	cvSetImageROI(stacked, cvRect(0, img1->height, img2->width, img2->height));
	cvAdd(img2, stacked, stacked, NULL);
	cvResetImageROI(stacked);

	return stacked;
}
void display_big_img(IplImage* img, char* title)
{
	IplImage* small;
	//GdkScreen* scr;
	int scr_width=1000, scr_height=1000;
	double img_aspect, scr_aspect, scale;

	///* determine screen size to see if image fits on screen */
	//gdk_init(NULL, NULL);
	//scr = gdk_screen_get_default();
	//scr_width = gdk_screen_get_width(scr);
	//scr_height = gdk_screen_get_height(scr);

	if (img->width >= 0.90 * scr_width || img->height >= 0.90 * scr_height)
	{
		img_aspect = (double)(img->width) / img->height;
		scr_aspect = (double)(scr_width) / scr_height;

		if (img_aspect > scr_aspect)
			scale = 0.90 * scr_width / img->width;
		else
			scale = 0.90 * scr_height / img->height;

		small = cvCreateImage(cvSize(img->width * scale, img->height * scale),
			img->depth, img->nChannels);
		cvResize(img, small, CV_INTER_AREA);
	}
	else
		small = cvCloneImage(img);

	cvNamedWindow(title, 1);
	cvShowImage(title, small);
	cvReleaseImage(&small);
}
int array_double(void** array, int n, int size)
{
	void* tmp;

	tmp = realloc(*array, 2 * n * size);
	if (!tmp)
	{
		fprintf(stderr, "Warning: unable to allocate memory in array_double(),"
			" %s line %d\n", __FILE__, __LINE__);
		if (*array)
			free(*array);
		*array = NULL;
		return 0;
	}
	*array = tmp;
	return n * 2;
}
void fatal_error(char* format, ...)
{
	va_list ap;

	fprintf(stderr, "Error: ");

	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	abort();
}
double dist_sq_2D(CvPoint2D64f p1, CvPoint2D64f p2)
{
	double x_diff = p1.x - p2.x;
	double y_diff = p1.y - p2.y;

	return x_diff * x_diff + y_diff * y_diff;
}
