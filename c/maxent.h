#ifndef   __INCLUDE_HIPPO_MAXENT_H
#define   __INCLUDE_HIPPO_MAXENT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "quark.h"

typedef quark_t maxent_dictionary_t;
typedef struct
{
    int    aid;                /**< Attribute id. */
    double  value;              /**< Value of the attribute. */
} maxent_attribute_t;
void maxent_attribute_init(maxent_attribute_t* cont);
void maxent_attribute_set(maxent_attribute_t* cont, int aid, double value);
void maxent_attribute_copy(maxent_attribute_t* dst, const maxent_attribute_t* src);
void maxent_attribute_swap(maxent_attribute_t* x, maxent_attribute_t* y);

typedef struct
{
    /** Number of items/labels in the sequence. */
    int         num_items;
    /** Maximum number of items/labels (internal use). */
    int         cap_items;
    /** Array of the item sequence. */
    maxent_attribute_t  *items;
    /** Array of the label sequence. */
    int         label;
    /** Instance weight. */
    int  weight;
    /** Group ID of the instance. */
    int         group;
} maxent_instance_t;
void maxent_instance_set_label(maxent_instance_t* inst, int label);
int  maxent_instance_append_attribute(maxent_instance_t* item, const maxent_attribute_t* cont);
void maxent_instance_destroy(maxent_instance_t* inst);
void maxent_instance_init(maxent_instance_t* inst);
void maxent_instance_init_n(maxent_instance_t* inst, int num_items);
void maxent_instance_copy(maxent_instance_t* dst, const maxent_instance_t* src);
int  maxent_instance_empty(maxent_instance_t* inst);
/**
 * A data set.
 *  A data set consists of an array of instances and dictionary objects
 *  for attributes and labels.
 */
typedef struct
{
    /** Number of instances. */
    int                 num_instances;
    /** Maximum number of instances (internal use). */
    int                 cap_instances;
    /** Array of instances. */
    maxent_instance_t*     instances;

    /** Dictionary object for attributes. */
    maxent_dictionary_t    *attrs;
    /** Dictionary object for labels. */
    maxent_dictionary_t    *labels;
} maxent_data_t;
maxent_data_t* maxent_data_create(void);
void maxent_data_init(maxent_data_t* data);
void maxent_data_init_n(maxent_data_t* data, int n);
void maxent_data_destroy(maxent_data_t* data);
int  maxent_data_append(maxent_data_t* data, const maxent_instance_t* inst);


maxent_dictionary_t*    maxent_dictionary_create(void);
int  maxent_dictionary_to_id(maxent_dictionary_t* dic, const char *str);
int  maxent_dictionary_to_string(maxent_dictionary_t* dic, int id, char **pstr);
int  maxent_dictionary_num(maxent_dictionary_t* dic);
int  maxent_dictionary_get(maxent_dictionary_t* dic, const char *str);
void maxent_dictionary_free(maxent_dictionary_t* dic, const char *str);
void maxent_dictionary_destroy(maxent_dictionary_t* dic);

typedef struct
{

    /**
     * Source id.
     *    The semantic of this field depends on the feature type:
     *    - attribute id for state features (type == 0).
     *    - output label id for transition features (type != 0).
     */
    int        src;

    /**
     * Destination id.
     *    Label id emitted by this feature.
     */
    int        dst;

    /**
     * Frequency (observation expectation).
     */
    int    freq;
    double weight;
} maxent_feature_t;
maxent_feature_t* maxent_feature_create(void);
void maxent_feature_init(maxent_feature_t* item);
void maxent_feature_destroy(maxent_feature_t *item);

typedef struct
{
    int     num_features;    /**< Number of features referred */
    int*    fids;            /**< Array of feature ids */
} feature_refs_t;
typedef struct
{
    int num_labels;                     ///标签总量        /**< Number of distinct output labels (L). */
    int num_attributes;                 ///属性总量        /**< Number of distinct attributes (A). */

    int num_instances;                  ///训练样本总量

    int num_features;                   ///特征总数    /**< Number of distinct features (K). */
    maxent_feature_t *features;         ///特征数组  /**< Array of feature descriptors [K]. */
    feature_refs_t* attributes;         ///特征数组索引  /**< References to attribute features [A]. */
    //int* freq;
    //feature_refs_t* forward_trans;  /**< References to transition features [L]. */
} maxent_features_model_t;
typedef struct
{
    maxent_instance_t*     instances;
    maxent_dictionary_t    *attrs;///
    maxent_dictionary_t    *labels;
    maxent_features_model_t *feats_model;
    unsigned char flag;
    int lbfgs_flag;
    double  reg2;
} maxent_model_t;
typedef struct
{
    char *_label;
    double _value;
}maxent_predict_item_;
typedef struct
{
    maxent_predict_item_ *items;
    int labels;
}maxent_predict_item;
maxent_predict_item* maxent_predict_item_create(int size_);
void maxent_predict_item_destroy(maxent_predict_item* item);
typedef struct {
    double  c1;                         ///Coefficient for L1 regularization.
    double  c2;                         ///Coefficient for L2 regularization.
    int     memory;                     ///The number of limited memories for approximating the inverse hessian matrix.
    double  epsilon;                    ///Epsilon for testing the convergence of the objective.
    int     stop;                       ///The duration of iterations to test the stopping criterion.
    double  delta;                      ///The threshold for the stopping criterion; an L-BFGS iteration stops when the
                                        ///improvement of the log likelihood over the last ${period} iterations is no
                                        ///greater than this threshold.
    int     max_iterations;             ///The maximum number of iterations for L-BFGS optimization.
    char*   linesearch;                 ///The line search algorithm used in L-BFGS updates:
                                        ///     'MoreThuente': More and Thuente's method,
                                        ///     'Backtracking': Backtracking method with regular Wolfe condition,
                                        ///     'StrongBacktracking': Backtracking method with strong Wolfe condition

    int     linesearch_max_iterations;  ///The maximum number of trials for the line search algorithm.

} lbfgs_training_option_t;
lbfgs_training_option_t* lbfgs_training_option_create(void);
maxent_model_t*         train(maxent_data_t* trainset,double c2);
int                     save(maxent_model_t* trainer,char* filename);
maxent_model_t*         load(char* filename);
maxent_predict_item*    predict(maxent_model_t* trainer,maxent_instance_t* instant);
void maxent_model_destroy(maxent_model_t* trainer);
void maxent_model_destroy_(maxent_model_t* trainer);
//maxent_data_t*          append(maxent_data_t* data,maxent_instance_t* instant);


void maxent_dictionary_destroy(maxent_dictionary_t* dic);
int maxent_dictionary_get(maxent_dictionary_t* dic, const char *str);
int maxent_dictionary_to_id(maxent_dictionary_t* dic, const char *str);
int maxent_dictionary_to_string(maxent_dictionary_t* dic, int id, char **pstr);
int maxent_dictionary_num(maxent_dictionary_t* dic);
void maxent_dictionary_free(maxent_dictionary_t* dic, const char *str);
maxent_dictionary_t* maxent_dictionary_create(void);

#endif
