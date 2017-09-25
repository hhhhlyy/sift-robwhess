#include "kdtree.h"
#include "minpq.h"
#include "imgfeatures.h"
#include "utils.h"

#include <cxcore.h>

#include <stdio.h>

struct bbf_data
{
	double d;
	void* old_data;
};

/************************* Local Function Prototypes *************************/

static struct kd_node* kd_node_init(struct feature*, int);
static void expand_kd_node_subtree(struct kd_node*);
static void assign_part_key(struct kd_node*);
static double median_select(double*, int);
static double rank_select(double*, int, int);
static void insertion_sort(double*, int);
static int partition_array(double*, int, double);
static void partition_features(struct kd_node*);
static struct kd_node* explore_to_leaf(struct kd_node*, struct feature*,
	struct min_pq*);
static int insert_into_nbr_array(struct feature*, struct feature**, int, int);
static int within_rect(CvPoint2D64f, CvRect);


/******************** Functions prototyped in keyptdb.h **********************/
struct kd_node* kdtree_build(struct feature* features, int n)
{
	struct kd_node* kd_root;

	if (!features || n <= 0)
	{
		fprintf(stderr, "Warning: kdtree_build(): no features, %s, line %d\n",
			__FILE__, __LINE__);
		return NULL;
	}

	kd_root = kd_node_init(features, n);
	expand_kd_node_subtree(kd_root);

	return kd_root;
}

int kdtree_bbf_knn(struct kd_node* kd_root, struct feature* feat, int k,
	struct feature*** nbrs, int max_nn_chks)
{
	struct kd_node* expl;
	struct min_pq* min_pq;
	struct feature* tree_feat, ** _nbrs;
	struct bbf_data* bbf_data;
	int i, t = 0, n = 0;

	if (!nbrs || !feat || !kd_root)
	{
		fprintf(stderr, "Warning: NULL pointer error, %s, line %d\n",
			__FILE__, __LINE__);
		return -1;
	}

	_nbrs = (feature**)calloc(k, sizeof(struct feature*));
	min_pq = minpq_init();
	minpq_insert(min_pq, kd_root, 0);
	while (min_pq->n > 0 && t < max_nn_chks)
	{
		expl = (struct kd_node*)minpq_extract_min(min_pq);
		if (!expl)
		{
			fprintf(stderr, "Warning: PQ unexpectedly empty, %s line %d\n",
				__FILE__, __LINE__);
			goto fail;
		}

		expl = explore_to_leaf(expl, feat, min_pq);
		if (!expl)
		{
			fprintf(stderr, "Warning: PQ unexpectedly empty, %s line %d\n",
				__FILE__, __LINE__);
			goto fail;
		}

		for (i = 0; i < expl->n; i++)
		{
			tree_feat = &expl->features[i];
			bbf_data = (struct bbf_data*)malloc(sizeof(struct bbf_data));
			if (!bbf_data)
			{
				fprintf(stderr, "Warning: unable to allocate memory,"
					" %s line %d\n", __FILE__, __LINE__);
				goto fail;
			}
			bbf_data->old_data = tree_feat->feature_data;
			bbf_data->d = descr_dist_sq(feat, tree_feat);
			tree_feat->feature_data = bbf_data;
			n += insert_into_nbr_array(tree_feat, _nbrs, n, k);
		}
		t++;
	}

	minpq_release(&min_pq);
	for (i = 0; i < n; i++)
	{
		bbf_data = (struct bbf_data*)_nbrs[i]->feature_data;
		_nbrs[i]->feature_data = bbf_data->old_data;
		free(bbf_data);
	}
	*nbrs = _nbrs;
	return n;

fail:
	minpq_release(&min_pq);
	for (i = 0; i < n; i++)
	{
		bbf_data = (struct bbf_data*)_nbrs[i]->feature_data;
		_nbrs[i]->feature_data = bbf_data->old_data;
		free(bbf_data);
	}
	free(_nbrs);
	*nbrs = NULL;
	return -1;
}

static struct kd_node* kd_node_init(struct feature* features, int n)
{
	struct kd_node* kd_node;

	kd_node = (struct kd_node*)malloc(sizeof(struct kd_node));
	memset(kd_node, 0, sizeof(struct kd_node));
	kd_node->ki = -1;
	kd_node->features = features;
	kd_node->n = n;

	return kd_node;
}

static void expand_kd_node_subtree(struct kd_node* kd_node)
{
	/* base case: leaf node */
	if (kd_node->n == 1 || kd_node->n == 0)
	{
		kd_node->leaf = 1;
		return;
	}

	assign_part_key(kd_node);
	partition_features(kd_node);

	if (kd_node->kd_left)
		expand_kd_node_subtree(kd_node->kd_left);
	if (kd_node->kd_right)
		expand_kd_node_subtree(kd_node->kd_right);
}

static struct kd_node* explore_to_leaf(struct kd_node* kd_node,
	struct feature* feat,
	struct min_pq* min_pq)
{
	struct kd_node* unexpl, *expl = kd_node;
	double kv;
	int ki;

	while (expl && !expl->leaf)
	{
		ki = expl->ki;
		kv = expl->kv;

		if (ki >= feat->d)
		{
			fprintf(stderr, "Warning: comparing imcompatible descriptors, %s" \
				" line %d\n", __FILE__, __LINE__);
			return NULL;
		}
		if (feat->descr[ki] <= kv)
		{
			unexpl = expl->kd_right;
			expl = expl->kd_left;
		}
		else
		{
			unexpl = expl->kd_left;
			expl = expl->kd_right;
		}

		if (minpq_insert(min_pq, unexpl, ABS(kv - feat->descr[ki])))
		{
			fprintf(stderr, "Warning: unable to insert into PQ, %s, line %d\n",
				__FILE__, __LINE__);
			return NULL;
		}
	}

	return expl;
}

void kdtree_release(struct kd_node* kd_root)
{
	if (!kd_root)
		return;
	kdtree_release(kd_root->kd_left);
	kdtree_release(kd_root->kd_right);
	free(kd_root);
}
static void assign_part_key(struct kd_node* kd_node)
{
	struct feature* features;
	double kv, x, mean, var, var_max = 0;
	double* tmp;
	int d, n, i, j, ki = 0;

	features = kd_node->features;
	n = kd_node->n;
	d = features[0].d;

	/* partition key index is that along which descriptors have most variance */
	for (j = 0; j < d; j++)
	{
		mean = var = 0;
		for (i = 0; i < n; i++)
			mean += features[i].descr[j];
		mean /= n;
		for (i = 0; i < n; i++)
		{
			x = features[i].descr[j] - mean;
			var += x * x;
		}
		var /= n;

		if (var > var_max)
		{
			ki = j;
			var_max = var;
		}
	}

	/* partition key value is median of descriptor values at ki */
	tmp = (double*)calloc(n, sizeof(double));
	for (i = 0; i < n; i++)
		tmp[i] = features[i].descr[ki];
	kv = median_select(tmp, n);
	free(tmp);

	kd_node->ki = ki;
	kd_node->kv = kv;
}
static void partition_features(struct kd_node* kd_node)
{
	struct feature* features, tmp;
	double kv;
	int n, ki, p, i, j = -1;

	features = kd_node->features;
	n = kd_node->n;
	ki = kd_node->ki;
	kv = kd_node->kv;
	for (i = 0; i < n; i++)
		if (features[i].descr[ki] <= kv)
		{
			tmp = features[++j];
			features[j] = features[i];
			features[i] = tmp;
			if (features[j].descr[ki] == kv)
				p = j;
		}
	tmp = features[p];
	features[p] = features[j];
	features[j] = tmp;

	/* if all records fall on same side of partition, make node a leaf */
	if (j == n - 1)
	{
		kd_node->leaf = 1;
		return;
	}

	kd_node->kd_left = kd_node_init(features, j + 1);
	kd_node->kd_right = kd_node_init(features + (j + 1), (n - j - 1));
}

static double median_select(double* array, int n)
{
	return rank_select(array, n, (n - 1) / 2);
}
static double rank_select(double* array, int n, int r)
{
	double* tmp, med;
	int gr_5, gr_tot, rem_elts, i, j;

	/* base case */
	if (n == 1)
		return array[0];

	/* divide array into groups of 5 and sort them */
	gr_5 = n / 5;
	gr_tot = cvCeil(n / 5.0);
	rem_elts = n % 5;
	tmp = array;
	for (i = 0; i < gr_5; i++)
	{
		insertion_sort(tmp, 5);
		tmp += 5;
	}
	insertion_sort(tmp, rem_elts);

	/* recursively find the median of the medians of the groups of 5 */
	tmp = (double*)calloc(gr_tot, sizeof(double));
	for (i = 0, j = 2; i < gr_5; i++, j += 5)
		tmp[i] = array[j];
	if (rem_elts)
		tmp[i++] = array[n - 1 - rem_elts / 2];
	med = rank_select(tmp, i, (i - 1) / 2);
	free(tmp);

	/* partition around median of medians and recursively select if necessary */
	j = partition_array(array, n, med);
	if (r == j)
		return med;
	else if (r < j)
		return rank_select(array, j, r);
	else
	{
		array += j + 1;
		return rank_select(array, (n - j - 1), (r - j - 1));
	}
}
static void insertion_sort(double* array, int n)
{
	double k;
	int i, j;

	for (i = 1; i < n; i++)
	{
		k = array[i];
		j = i - 1;
		while (j >= 0 && array[j] > k)
		{
			array[j + 1] = array[j];
			j -= 1;
		}
		array[j + 1] = k;
	}
}
static int partition_array(double* array, int n, double pivot)
{
	double tmp;
	int p, i, j;

	i = -1;
	for (j = 0; j < n; j++)
		if (array[j] <= pivot)
		{
			tmp = array[++i];
			array[i] = array[j];
			array[j] = tmp;
			if (array[i] == pivot)
				p = i;
		}
	array[p] = array[i];
	array[i] = pivot;

	return i;
}
static int insert_into_nbr_array(struct feature* feat, struct feature** nbrs,
	int n, int k)
{
	struct bbf_data* fdata, *ndata;
	double dn, df;
	int i, ret = 0;

	if (n == 0)
	{
		nbrs[0] = feat;
		return 1;
	}

	/* check at end of array */
	fdata = (struct bbf_data*)feat->feature_data;
	df = fdata->d;
	ndata = (struct bbf_data*)nbrs[n - 1]->feature_data;
	dn = ndata->d;
	if (df >= dn)
	{
		if (n == k)
		{
			feat->feature_data = fdata->old_data;
			free(fdata);
			return 0;
		}
		nbrs[n] = feat;
		return 1;
	}

	/* find the right place in the array */
	if (n < k)
	{
		nbrs[n] = nbrs[n - 1];
		ret = 1;
	}
	else
	{
		nbrs[n - 1]->feature_data = ndata->old_data;
		free(ndata);
	}
	i = n - 2;
	while (i >= 0)
	{
		ndata = (struct bbf_data*)nbrs[i]->feature_data;
		dn = ndata->d;
		if (dn <= df)
			break;
		nbrs[i + 1] = nbrs[i];
		i--;
	}
	i++;
	nbrs[i] = feat;

	return ret;
}