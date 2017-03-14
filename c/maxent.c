#include "maxent.h"
#include "lbfgs.h"
#include "rumavl.h"
#include "tpl.h"
#include "cqdb.h"
#define    COMP(a, b)    ((a)>(b))-((a)<(b))
#define LBFGS_ERROR 0x01
#define LBFGS_NORMAL 0xFF
/**
 * Feature set.
 */
typedef struct
{
    RUMAVL* avl;    /**< Root node of the AVL tree. */
    int num;        /**< Number of features in the AVL tree. */
} maxent_feature_set_t;


static int feature_set_comp(const void *x, const void *y, size_t n, void *udata)
{
    int ret = 0;
    const maxent_feature_t* f1 = (const maxent_feature_t*)x;
    const maxent_feature_t* f2 = (const maxent_feature_t*)y;

    ret = COMP(f1->src, f2->src);
    if (ret == 0)
    {
        ret = COMP(f1->dst, f2->dst);
    }
    return ret;
}

static maxent_feature_set_t* maxent_feature_set_create(void)
{
    maxent_feature_set_t* set = NULL;
    set = (maxent_feature_set_t*)calloc(1, sizeof(maxent_feature_set_t));
    if (set != NULL)
    {
        set->num = 0;
        set->avl = rumavl_new(
                       sizeof(maxent_feature_t), feature_set_comp, NULL, NULL);
        if (set->avl == NULL)
        {
            free(set);
            set = NULL;
        }
    }
    return set;
}

static void feature_set_delete(maxent_feature_set_t* set)
{
    if (set != NULL)
    {
        rumavl_destroy(set->avl);
        free(set);
    }
}
static int feature_set_add(maxent_feature_set_t* set, const maxent_feature_t* f)
{
    /* Check whether if the feature already exists. */
    maxent_feature_t *p = (maxent_feature_t*)rumavl_find(set->avl, f);
    if (p == NULL)
    {
        /* Insert the feature to the feature set. */
        rumavl_insert(set->avl, f);
        ++set->num;
    }
    else
    {
        /* An existing feature: add the observation expectation. */
        p->freq += f->freq;
    }
    return 0;
}

static maxent_feature_t* feature_set_generate(int *ptr_num_features,maxent_feature_set_t* set)
{
    int n = 0, k = 0;
    RUMAVL_NODE *node = NULL;
    maxent_feature_t *f = NULL;
    maxent_feature_t *features = NULL;

    /* The first pass: count the number of valid features. */
    while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL)
        ++n;

    /* The second path: copy the valid features to the feature array. */
    features = (maxent_feature_t*)calloc(n, sizeof(maxent_feature_t));
    if (features != NULL)
    {
        node = NULL;
        while ((node = rumavl_node_next(set->avl, node, 1, (void**)&f)) != NULL)
        {
            memcpy(&features[k], f, sizeof(maxent_feature_t));
            ++k;
        }
        *ptr_num_features = n;
        return features;
    }
    else
    {
        *ptr_num_features = 0;
        return NULL;
    }
}
void maxent_model_destroy(maxent_model_t* _model)
{
    int i;
    maxent_instance_t *inst=NULL;
    if(_model->instances!=NULL)
    {
        for (i = 0; i < _model->feats_model->num_instances; ++i)
        {
            inst=&_model->instances[i];
            if(inst!=NULL&&inst->items!=NULL)
                free(inst->items);
        }
        free(_model->instances);
    }
    _model->instances=NULL;
    maxent_dictionary_destroy(_model->attrs);
    _model->attrs=NULL;
    maxent_dictionary_destroy(_model->labels);
    _model->labels=NULL;
    if(_model->feats_model==NULL)
    {
        free(_model);
        return;
    }

    if(_model->feats_model->features!=NULL)free(_model->feats_model->features);
    for(i=0; i<_model->feats_model->num_attributes; i++)
    {
        if(_model->feats_model->attributes[i].num_features>0)
            free(_model->feats_model->attributes[i].fids);
    }
    if(_model->feats_model->attributes!=NULL)free(_model->feats_model->attributes);
    free(_model->feats_model);
    free(_model);
}
void maxent_model_destroy_(maxent_model_t* _model)
{
    int i;
    _model->instances=NULL;
    _model->attrs=NULL;
    _model->labels=NULL;
    if(_model->feats_model==NULL)
    {
        free(_model);
        return;
    }

    if(_model->feats_model->features!=NULL)free(_model->feats_model->features);
    for(i=0; i<_model->feats_model->num_attributes; i++)
    {
        if(_model->feats_model->attributes[i].num_features>0)
            free(_model->feats_model->attributes[i].fids);
    }
    if(_model->feats_model->attributes!=NULL)free(_model->feats_model->attributes);
    free(_model->feats_model);
    free(_model);
}
maxent_feature_t* maxent_feature_create(void)
{
    maxent_feature_t* features=NULL;
    features=(maxent_feature_t*)malloc(sizeof(maxent_feature_t));
    if(features==NULL) return features;
    return features;
}
void maxent_feature_init(maxent_feature_t* features)
{
    features->dst=-1;
    features->src=-1;
    features->freq=1;
    features->weight=0.0;
}

void maxent_feature_destroy(maxent_feature_t *item)
{
    free(item);
}
static maxent_feature_t* generate_feature_arrays(int *ptr_num_features,maxent_data_t* trainset)
{

    int i,j;
    maxent_feature_t *features = NULL,*re=NULL;
    maxent_feature_set_t* set = NULL;
    maxent_instance_t * _instance=NULL;
    maxent_attribute_t* _attri=NULL;
    const int N = trainset->num_instances;


    set = maxent_feature_set_create();
    features=maxent_feature_create();

    for (i = 0; i < N; ++i)
    {
        ///迭代trainset
        ///单样本
        _instance=&trainset->instances[i];

        ///------------------针对单样本计算P(Y|x)------------
        for(j=0; j<_instance->num_items; j++)
        {
            ///迭代单样本_instance中的属性maxent_attribute_t:_attri
            maxent_feature_init(features);
            _attri=&_instance->items[j];
            features->dst=_instance->label;
            features->src=_attri->aid;
            features->freq=_instance->weight;
            feature_set_add(set, features);
        }
    }


    /* Convert the feature set to an feature array. */
    /***操作AVL树，得到filter后的特征数组列表**/
    re = feature_set_generate(ptr_num_features, set);

    /* Delete the feature set. */
    feature_set_delete(set);

    return re;
}

static feature_refs_t* generate_feature_index(maxent_feature_t *features,int num_features,int num_attributes)
{
    int i, k;
    feature_refs_t *fl_ = NULL;
    feature_refs_t *attributes = NULL;
    maxent_feature_t* feats=NULL;
///re->attributes=generate_feature_index(re->features,re->num_features,re->num_attributes);
    /***根据属性总数，构造A长度的数组****/
    attributes = (feature_refs_t*)calloc(num_attributes, sizeof(feature_refs_t));
    if (attributes == NULL) goto error_exit;

    /*
        Firstly, loop over the features to count the number of references.
        We don't use realloc() to avoid memory fragmentation.
     */
    for (k = 0; k < num_features; ++k)
    {
        /***迭代所有的特征，根据其属性type，起点src，初始化子数组的长度****/
        feats = &features[k];
        attributes[feats->src].num_features++;
    }

    /*
        Secondarily, allocate memory blocks to store the feature references.
        We also clear fl->num_features fields, which will be used as indices
        in the next phase.
     */
    for (i = 0; i < num_attributes; ++i)
    {
        /***属性数组操作:根据子数组的长度，创建对应长度的int数组****/

        fl_ = &attributes[i];
        if(fl_->num_features>0)
        {
            fl_->fids = (int*)calloc(fl_->num_features, sizeof(int));
            if (fl_->fids == NULL) goto error_exit;
        }
        else
            fl_->fids=NULL;
        fl_->num_features = 0;///为了下一步填充数组，将其作为位置索引
    }
    /*
        Finally, store the feature indices.
     */
    for (k = 0; k < num_features; ++k)
    {
        /***迭代所有的特征，根据其属性type，起点src，设置子数组的值(索引，为其在特征数组中的下标)****/
        feats = &features[k];
        /*******/
        fl_ = &attributes[feats->src];
        fl_->fids[fl_->num_features++] = k;
    }
    return attributes;

error_exit:
    if (attributes != NULL)
    {
        for (i = 0; i < num_attributes; ++i) free(attributes[i].fids);
        free(attributes);
    }
    return NULL;
}
static maxent_features_model_t* init_feats_model(maxent_data_t* trainset)
{
    maxent_features_model_t* re=NULL;
    re=(maxent_features_model_t*)malloc(sizeof(maxent_features_model_t));
    re->num_instances=trainset->num_instances;
    re->num_labels=maxent_dictionary_num(trainset->labels);
    re->num_attributes=maxent_dictionary_num(trainset->attrs);
    re->features=generate_feature_arrays(&re->num_features,trainset);
    re->attributes=generate_feature_index(re->features,re->num_features,re->num_attributes);
    return re;
}
enum
{
    WSTATE_NONE,
    WSTATE_LABELS,
    WSTATE_ATTRS,
    WSTATE_LABELREFS,
    WSTATE_ATTRREFS,
    WSTATE_FEATURES,
};
typedef struct
{
    uint8_t     magic[6];       /* File magic. */
    uint32_t    size;           /* File size. */
    uint32_t    version;        /* Version number. */
    uint32_t    num_features;   /* Number of features. */
    uint32_t    num_labels;     /* Number of labels. */
    uint32_t    num_attributes;      /* Number of attributes. */
    uint32_t    num_instances;
    uint8_t     flag;
    uint32_t    off_options;
    uint32_t    off_features;   /* Offset to features. */
    uint32_t    off_labels;     /* Offset to label CQDB. */
    uint32_t    off_attrs;      /* Offset to attribute CQDB. */
    uint32_t    off_attrrefs;   /* Offset to attribute feature references. */
} header_t;
typedef struct
{
    uint8_t     chunk[4];       /* Chunk id */
    uint32_t    size;           /* Chunk size. */
    uint32_t    num;            /* Number of items. */
    uint32_t    offsets[1];     /* Offsets. */
} featureref_header_t;

typedef struct
{
    uint8_t     chunk[4];       /* Chunk id */
    uint32_t    size;           /* Chunk size. */
    uint32_t    num;            /* Number of items. */
} feature_header_t;
typedef struct
{
    FILE *fp;
    int state;
    header_t header;
    cqdb_writer_t* dbw;
    featureref_header_t* href;
    feature_header_t* hfeat;
} maxent_writer_t;
#define FILEMAGIC       "MaxEnt"
#define VERSION_NUMBER  (001)
#define CHUNK_ATTRREF   "ATRF"
#define CHUNK_FEATURE   "FEAT"
#define HEADER_SIZE     sizeof(header_t)
#define CHUNK_SIZE      sizeof(feature_header_t) ///feature list
static void maxent_writer_close_(maxent_writer_t* writer)
{
    if (writer != NULL)
    {
        if (writer->fp != NULL)
        {
            fclose(writer->fp);
        }
        free(writer);
    }
}
static int maxent_writer_close(maxent_writer_t* writer)
{
    FILE *fp = writer->fp;
    header_t *header = &writer->header;

    /* Store the file size. */
    header->size = (uint32_t)ftell(fp);

    /* Move the file position to the head. */
    if (fseek(fp, 0, SEEK_SET) != 0)
    {
        goto error_exit;
    }
    /* Write the file header. */
    write_uint8_array(fp, header->magic, sizeof(header->magic));
    write_uint32(fp, header->size);
    write_uint32(fp, header->version);
    write_uint32(fp, header->num_features);
    write_uint32(fp, header->num_labels);
    write_uint32(fp, header->num_attributes);
    write_uint32(fp, header->num_instances);
    write_uint8(fp, header->flag);
    write_uint32(fp, header->off_options);
    write_uint32(fp, header->off_features);
    write_uint32(fp, header->off_labels);
    write_uint32(fp, header->off_attrs);
    write_uint32(fp, header->off_attrrefs);

    /* Check for any error occurrence. */
    if (ferror(fp))
    {
        goto error_exit;
    }

    /* Close the writer. */
    fclose(fp);
    free(writer);
    return 0;

error_exit:
    if (writer != NULL)
    {
        if (writer->fp != NULL)
        {
            fclose(writer->fp);
        }
        free(writer);
    }
    return 1;
}
static maxent_writer_t* maxent_writer_create(const char *filename)
{
    header_t *header = NULL;
    maxent_writer_t *writer = NULL;

    /* Create a writer instance. */
    writer = (maxent_writer_t*)calloc(1, sizeof(maxent_writer_t));///创建"写"对象
    if (writer == NULL) return writer;

    /* Open the file for writing. */
    writer->fp = fopen(filename, "wb");///write读写的内置文件指针
    if (writer->fp == NULL)
    {
        free(writer);
        return NULL;
    }

    /* Fill the members in the header. */
    header = &writer->header;///write对象的文件头
    strncpy((char*)header->magic, FILEMAGIC, 6);///设置文件头的指定属性
    header->version = VERSION_NUMBER;

    /* Advance the file position to skip the file header. */
    if (fseek(writer->fp, HEADER_SIZE, SEEK_CUR) != 0)  ///文件读写位置指针从当前位置后移48字节
    {
        if (writer != NULL)
        {
            if (writer->fp != NULL)
            {
                fclose(writer->fp);
            }
            free(writer);
        }
        return NULL;
    }

    return writer;
}
static int maxent_writer_features_open(maxent_writer_t* writer,int num_features)
{
    FILE *fp = writer->fp;
    feature_header_t* hfeat = NULL;

    /* Check if we aren't writing anything at this moment. */
    if (writer->state != WSTATE_NONE)  ///检查写对象的状态，
        return -1;


    /* Allocate a feature chunk header. */
    hfeat = (feature_header_t*)calloc(sizeof(feature_header_t), 1);
    if (hfeat == NULL)
        return -1;

    writer->header.off_features = (uint32_t)ftell(fp);///特征对象块在目标文件中的起始位置
    fseek(fp, CHUNK_SIZE, SEEK_CUR);///指针后移12字节，为feature_header_t预留位置

    strncpy((char*)hfeat->chunk, CHUNK_FEATURE, 4);
    hfeat->num=0;
    writer->hfeat = hfeat;

    writer->state = WSTATE_FEATURES;///设置写对象状态
    writer->header.num_features=num_features;
    return 0;
}
static int maxent_writer_features_put(maxent_writer_t* writer, const maxent_feature_t* f)
{
    FILE *fp = writer->fp;
    feature_header_t* hfeat = writer->hfeat;

    /* Make sure that we are writing attribute feature references. */
    if (writer->state != WSTATE_FEATURES)
    {
        return -1;
    }

    write_uint32(fp, f->freq);
    write_uint32(fp, f->src);
    write_uint32(fp, f->dst);
    write_double(fp, f->weight);
    ++hfeat->num;
    return 0;
}
static int maxent_writer_features_close(maxent_writer_t* writer)
{
    FILE *fp = writer->fp;
    feature_header_t* hfeat = writer->hfeat;
    uint32_t begin = writer->header.off_features, end = 0;

    /* Make sure that we are writing attribute feature references. */
    if (writer->state != WSTATE_FEATURES)
    {
        return -1;
    }

    /* Store the current offset position. */
    end = (uint32_t)ftell(fp);

    /* Compute the size of this chunk. */
    /***
    begin 保存特征数组的文件起始位置
    end   保存特征数组的文件的结束位置
    ***/
    hfeat->size = (end - begin);

    /* Write the chunk header and offset array. */
    fseek(fp, begin, SEEK_SET);///指针移动到起始位置,然后写特征数组文件的文件头
    write_uint8_array(fp, hfeat->chunk, 4);
    write_uint32(fp, hfeat->size);
    write_uint32(fp, hfeat->num);

    /* Move the file pointer to the tail. */
    fseek(fp, end, SEEK_SET);///指针移动到结束位置，开始下一步的写文件

    /* Uninitialize. */
    free(hfeat);
    writer->hfeat = NULL;
    writer->state = WSTATE_NONE;///设置写文件状态标示
    return 0;
}

/***
"写"标签对象的准备工作：
参数:
num_labels      标签对象个数
***/
static int maxent_writer_labels_open(maxent_writer_t* writer, int num_labels)
{
    /* Check if we aren't writing anything at this moment. */
    if (writer->state != WSTATE_NONE)
    {
        return 1;
    }

    /* Store the current offset. */
    writer->header.off_labels = (uint32_t)ftell(writer->fp);

    /* Open a CQDB chunk for writing. */
    writer->dbw = cqdb_writer(writer->fp, 0);
    if (writer->dbw == NULL)
    {
        writer->header.off_labels = 0;
        return 1;
    }

    writer->state = WSTATE_LABELS;
    writer->header.num_labels = num_labels;
    return 0;
}

static int maxent_writer_labels_close(maxent_writer_t* writer)
{
    /* Make sure that we are writing labels. */
    if (writer->state != WSTATE_LABELS)
    {
        return 1;
    }

    /* Close the CQDB chunk. */
    if (cqdb_writer_close(writer->dbw))
    {
        return 1;
    }

    writer->dbw = NULL;
    writer->state = WSTATE_NONE;
    return 0;
}
/***
保存标签对象:将单个的标签对象保存到磁盘
参数:
writer          "写对象"的封装
lid             标签对象的id
value           标签对象的value
***/
static int maxent_writer_labels_put(maxent_writer_t* writer, int lid, const char *value)
{
    /* Make sure that we are writing labels. */
    if (writer->state != WSTATE_LABELS)
    {
        return 1;
    }

    /* Put the label. */
    if (cqdb_writer_put(writer->dbw, value, lid))
    {
        return 1;
    }

    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////
static int maxent_writer_attrs_open(maxent_writer_t* writer, int num_attrs)
{
    /* Check if we aren't writing anything at this moment. */
    if (writer->state != WSTATE_NONE)
    {
        return 1;
    }

    /* Store the current offset. */
    writer->header.off_attrs = (uint32_t)ftell(writer->fp);

    /* Open a CQDB chunk for writing. */
    writer->dbw = cqdb_writer(writer->fp, 0);
    if (writer->dbw == NULL)
    {
        writer->header.off_attrs = 0;
        return 1;
    }

    writer->state = WSTATE_ATTRS;
    writer->header.num_attributes = num_attrs;
    return 0;
}

static int maxent_writer_attrs_close(maxent_writer_t* writer)
{
    /* Make sure that we are writing attributes. */
    if (writer->state != WSTATE_ATTRS)
    {
        return 1;
    }

    /* Close the CQDB chunk. */
    if (cqdb_writer_close(writer->dbw))
    {
        return 1;
    }

    writer->dbw = NULL;
    writer->state = WSTATE_NONE;
    return 0;
}

static int maxent_writer_attrs_put(maxent_writer_t* writer, int aid, const char *value)
{
    /* Make sure that we are writing labels. */
    if (writer->state != WSTATE_ATTRS)
    {
        return 1;
    }

    /* Put the attribute. */
    if (cqdb_writer_put(writer->dbw, value, aid))
    {
        return 1;
    }

    return 0;
}
/***
"写"转换特征索引的准备工作：
参数:
num_labels          标签对象个数+2
***/


static int maxent_writer_attrrefs_open(maxent_writer_t* writer, int num_attrs)
{
    uint32_t offset;
    FILE *fp = writer->fp;
    featureref_header_t* href = NULL;
    size_t size = CHUNK_SIZE + sizeof(uint32_t) * num_attrs;

    /* Check if we aren't writing anything at this moment. */
    if (writer->state != WSTATE_NONE)
    {
        return -1;
    }

    /* Allocate a feature reference array. */
    href = (featureref_header_t*)calloc(size, 1);
    if (href == NULL)
    {
        return -1;
    }

    /* Align the offset to a DWORD boundary. */
    offset = (uint32_t)ftell(fp);
    while (offset % 4 != 0)
    {
        uint8_t c = 0;
        fwrite(&c, sizeof(uint8_t), 1, fp);
        ++offset;
    }

    /* Store the current offset position to the file header. */
    writer->header.off_attrrefs = offset;
    fseek(fp, size, SEEK_CUR);

    /* Fill members in the feature reference header. */
    strncpy((char*)href->chunk, CHUNK_ATTRREF, 4);
    href->size = 0;
    href->num = num_attrs;

    writer->href = href;
    writer->state = WSTATE_ATTRREFS;
    return 0;
}

static int maxent_writer_attrrefs_close(maxent_writer_t* writer)
{
    uint32_t i;
    FILE *fp = writer->fp;
    featureref_header_t* href = writer->href;
    uint32_t begin = writer->header.off_attrrefs, end = 0;

    /* Make sure that we are writing attribute feature references. */
    if (writer->state != WSTATE_ATTRREFS)
    {
        return -1;
    }

    /* Store the current offset position. */
    end = (uint32_t)ftell(fp);

    /* Compute the size of this chunk. */
    href->size = (end - begin);

    /* Write the chunk header and offset array. */
    fseek(fp, begin, SEEK_SET);
    write_uint8_array(fp, href->chunk, 4);
    write_uint32(fp, href->size);
    write_uint32(fp, href->num);
    for (i = 0; i < href->num; ++i)
    {
        write_uint32(fp, href->offsets[i]);
    }

    /* Move the file pointer to the tail. */
    fseek(fp, end, SEEK_SET);

    /* Uninitialize. */
    free(href);
    writer->href = NULL;
    writer->state = WSTATE_NONE;
    return 0;
}

static int maxent_writer_attrrefs_put(maxent_writer_t* writer, int aid, const feature_refs_t* ref)
{
    int i, fid;
    //uint32_t n = 0;
    FILE *fp = writer->fp;
    featureref_header_t* href = writer->href;

    /* Make sure that we are writing attribute feature references. */
    if (writer->state != WSTATE_ATTRREFS)
    {
        return -1;
    }

    /* Store the current offset to the offset array. */
    href->offsets[aid] = ftell(fp);

    /* Count the number of references to active features. */


    /* Write the feature reference. */
    write_uint32(fp, (uint32_t)ref->num_features);
    for (i = 0; i < ref->num_features; ++i)
    {
        fid = ref->fids[i];
        write_uint32(fp, (uint32_t)fid);
    }

    return 0;
}

static int maxent_writer_options(maxent_writer_t* writer,maxent_model_t* trainer)
{
    FILE *fp=NULL;
    fp=writer->fp;
    writer->header.off_options=(uint32_t)ftell(fp);
    write_double(fp,trainer->reg2);
    write_uint32(fp,trainer->lbfgs_flag);

    writer->header.num_instances=trainer->feats_model->num_instances;
    //writer->header.reg2=trainer->reg2;
    writer->header.flag=trainer->flag;
    //printf("save ......%d %d\n",trainer->flag,writer->header.flag);
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////
int save(maxent_model_t* trainer,char* filename)
{
    int i,ret;
    //int ret;
    maxent_writer_t *writer_=NULL;
    //maxent_features_model_t *model_internal=NULL;
    maxent_feature_t*   feature_item=NULL;
    feature_refs_t*     feature_refs_item=NULL;
    char * str=NULL;


    writer_=maxent_writer_create(filename);
    if(writer_==NULL) return -1;
    maxent_writer_options(writer_,trainer);

    ret=maxent_writer_features_open(writer_,trainer->feats_model->num_features);
    if(ret) goto error;
    for(i=0; i<trainer->feats_model->num_features; i++)
    {
        feature_item=&trainer->feats_model->features[i];
        ret=maxent_writer_features_put(writer_,feature_item);
        if(ret) goto error;
    }
    ret=maxent_writer_features_close(writer_);
    if(ret) goto error;

    ret=maxent_writer_labels_open(writer_,trainer->feats_model->num_labels);
    if(ret) goto error;
    for(i=0; i<trainer->feats_model->num_labels; i++)
    {
        maxent_dictionary_to_string(trainer->labels,i,&str);
        maxent_writer_labels_put(writer_,i,str);
        maxent_dictionary_free(trainer->labels,str);
    }
    ret=maxent_writer_labels_close(writer_);
    if(ret) goto error;
    ret=maxent_writer_attrs_open(writer_,trainer->feats_model->num_attributes);
    if(ret) goto error;
    for(i=0; i<trainer->feats_model->num_attributes; i++)
    {
        maxent_dictionary_to_string(trainer->attrs,i,&str);
        maxent_writer_attrs_put(writer_,i,str);
        maxent_dictionary_free(trainer->labels,str);
    }
    ret=maxent_writer_attrs_close(writer_);
    if(ret) goto error;

    ret=maxent_writer_attrrefs_open(writer_,trainer->feats_model->num_attributes);
    if(ret) goto error;
    for(i=0; i<trainer->feats_model->num_attributes; i++)
    {
        feature_refs_item=&trainer->feats_model->attributes[i];
        ret=maxent_writer_attrrefs_put(writer_,i,feature_refs_item);
        if(ret) goto error;
    }
    ret=maxent_writer_attrrefs_close(writer_);
    if(ret) goto error;

    ret=maxent_writer_close(writer_);
    if(ret) goto error;
    return 0;
error:
    maxent_writer_close_(writer_);
    return -1;
}

static feature_refs_t* maxent_read_feature_ref(uint8_t* buffer,header_t* header,int num_attrs)
{
    feature_refs_t *re=NULL;
    size_t size = CHUNK_SIZE + sizeof(uint32_t) * num_attrs;
    featureref_header_t *href=NULL;
    int i,j,num_features_=0,fid=-1;
    uint8_t *p=buffer+header->off_attrrefs,*q=NULL;

    href=(featureref_header_t*)calloc(size,1);
    if(href==NULL) return NULL;
    re=(feature_refs_t*)calloc(num_attrs,sizeof(feature_refs_t));
    if(re==NULL) return NULL;

    p += read_uint8_array(p, href->chunk, sizeof(href->chunk));
    p += read_uint32(p, &href->size);
    p += read_uint32(p, &href->num);
    for(i=0; i<num_attrs; i++)
    {
        p += read_uint32(p, (uint32_t*)(&href->offsets[i]));
        q=buffer+href->offsets[i];
        q += read_uint32(q, (uint32_t*)(&num_features_));
        if(num_features_<=0)
        {
            re[i].num_features=0;
            re[i].fids=NULL;
            continue;
        }
        re[i].num_features=num_features_;
        re[i].fids=(int*)calloc(num_features_,sizeof(int));///异常处理？？？？？？？
        if(re[i].fids==NULL)
        {
            for(j=0; j<i; j++)
                free(re[j].fids);
            free(re);
            free(href);
            return NULL;
        }
        for(j=0; j<num_features_; j++)
        {
            q += read_uint32(q, (uint32_t*)(&fid));
            re[i].fids[j]=fid;
        }
    }
    free(href);

    return re;
}
static maxent_feature_t* maxent_read_features(uint8_t* buffer,header_t* header)
{
    maxent_feature_t* re=NULL,*item=NULL;
    feature_header_t* hfeat=NULL;
    int i;
    uint8_t *p=buffer+header->off_features;
    hfeat=(feature_header_t*)malloc(sizeof(feature_header_t));
    if(hfeat==NULL) return NULL;
    p += read_uint8_array(p, hfeat->chunk, sizeof(hfeat->chunk));
    p += read_uint32(p, &hfeat->size);
    p += read_uint32(p, &hfeat->num);

    re=(maxent_feature_t*)calloc(hfeat->num,sizeof(maxent_feature_t));
    if(re==NULL) return NULL;
    for(i=0; i<hfeat->num; i++)
    {
        item=&re[i];
        p += read_uint32(p, (uint32_t*)(&item->freq));
        p += read_uint32(p, (uint32_t*)(&item->src));
        p += read_uint32(p, (uint32_t*)(&item->dst));
        p += read_double(p, &item->weight);
    }
    free(hfeat);
    return re;
}
static void maxent_model_init_header(maxent_model_t* re,header_t *header)
{
    re->feats_model->num_attributes=header->num_attributes;
    re->feats_model->num_features=header->num_features;
    re->feats_model->num_instances=header->num_instances;
    re->feats_model->num_labels=header->num_labels;
    re->flag=header->flag;
}
///返回值:目标函数值
static double maxent_objective_and_gradients_batch(
    maxent_model_t *model_data,             ///数据
    const lbfgsfloatval_t *lambda,          ///当前的参数估计值
    double *g)
{
    int num_labels_,i,j,k,fid,dest,N;
    double f=0.0,sum_=0.0;
    double *re=NULL;
    //maxent_predict_item *re=NULL;
    maxent_attribute_t* _attri=NULL;
    maxent_instance_t*  _instance=NULL;

    feature_refs_t* attr_feat_ref=NULL,*attr_feat_ref_item=NULL;
    //maxent_feature_t * feature_=NULL;

    N=model_data->feats_model->num_instances;
    num_labels_=model_data->feats_model->num_labels;///标签个数
    re =(double*)calloc(num_labels_,sizeof(double));///标签个数长度数组,元素值为sum(wi*fi(x1,y1)) sum(wi*fi(x1,y2))...
    if(re==NULL)
    {
        model_data->flag=LBFGS_ERROR;
        return f;
    }
    //re_=(double*)calloc(num_labels_,sizeof(double));///
    for(i=0; i<num_labels_; i++)///初始化为0
        re[i]=0.0;
    for(i=0; i<model_data->feats_model->num_features; i++) ///迭代特征数组,初始化梯度的经验分布
        g[i]=(-1.0)*model_data->feats_model->features[i].freq/N;

    attr_feat_ref=model_data->feats_model->attributes;
    for(i=0; i<N; i++)
    {
        ///迭代trainset
        ///单样本
        _instance=&model_data->instances[i];
        //_instance->label
        ///------------------针对单样本计算P(Y|x)------------
        for(k=0; k<_instance->num_items; k++)
        {
            ///迭代单样本_instance中的属性maxent_attribute_t:_attri
            _attri=&_instance->items[k];
            attr_feat_ref_item=&attr_feat_ref[_attri->aid];
            for(j=0; j<attr_feat_ref_item->num_features; j++)
            {
                fid=attr_feat_ref_item->fids[j];
                dest=model_data->feats_model->features[fid].dst;
                re[dest]+=lambda[fid];
            }
        }
        ///------------------针对单样本计算P(Y|x)------------
        for(k=0; k<num_labels_; k++)
        {
            //f-=(1.0/N)*re[k];
            ///  f: -P(X,Yj)* sum(wi*fi(X,Yj))  P(X,Yj)~1/N
            sum_+=exp(re[k]);
        }
        f-=(1.0/N)*re[_instance->label];
        f+=(1.0/N)*log(sum_);///  f: P(X) *Log( sum( exp(...) ) ) P(X)~1/N
        ///---------------更新梯度----------------------------
        for(k=0; k<_instance->num_items; k++)
        {
            ///迭代单样本_instance中的属性maxent_attribute_t:_attri
            _attri=&_instance->items[k];
            attr_feat_ref_item=&attr_feat_ref[_attri->aid];
            for(j=0; j<attr_feat_ref_item->num_features; j++)
            {
                fid=attr_feat_ref_item->fids[j];
                dest=model_data->feats_model->features[fid].dst;
                g[fid]+=(1.0/N)*exp(re[dest])/sum_;/// g: P(X)*Pw(Y|X)
            }
        }
        ///---------------更新梯度----------------------------
        for(k=0; k<num_labels_; k++)
        {
            re[k]=0.0;
        }
        sum_=0.0;
        ////////////////////////////////////////////////////////////////////////////

    }
    free(re);

    return f;

}
/**回调函数：提供梯度(g)和目标函数值(返回值)**/
static lbfgsfloatval_t lbfgs_evaluate(
    void *instance,           ///用户定义数据
    const lbfgsfloatval_t *x, ///当前的x
    lbfgsfloatval_t *g,       ///梯度
    const int n,              ///x变量个数
    const lbfgsfloatval_t step///
)
{

    int i;
    double f, norm = 0.0,c22;
    maxent_model_t *model_data = (maxent_model_t*)instance;

    /* Compute the objective value and gradients. */
    f=maxent_objective_and_gradients_batch(model_data,x,g);

    /* L2 regularization. */

    if (0 < model_data->reg2)
    {
        c22 = model_data->reg2 * 2.;
        for (i = 0; i < n; ++i)
        {
            g[i] += (c22 * x[i]);///g+2x
            norm += x[i] * x[i];
        }
        f += (model_data->reg2 * norm);///目标函数值+    ||X||2
    }

    return f;
}
/**回调函数：迭代进程中，供用户使用 返回0-继续，非0-终止lbfgs**/
static int lbfgs_progress(
    void *instance,const lbfgsfloatval_t *x,const lbfgsfloatval_t *g,const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,const lbfgsfloatval_t gnorm,const lbfgsfloatval_t step,
    int n,int k,int ls)
{
    int i;
    maxent_model_t *model_data = (maxent_model_t*)instance;
    if(model_data->flag==LBFGS_ERROR) return 1;
    /* Store the feature weight in case L-BFGS terminates with an error. */
    for (i = 0; i < n; ++i)
    {
        model_data->feats_model->features[i].weight=x[i];
        //model_data->lambda[i]= x[i];
    }
    /* Continue. */

    return 0;
}
static void lbfgs_parameter_update(lbfgs_parameter_t* lbfgs,lbfgs_training_option_t* opt)
{
    lbfgs->m = opt->memory;
    lbfgs->epsilon = opt->epsilon;
    lbfgs->past = opt->stop;
    lbfgs->delta = opt->delta;
    lbfgs->max_iterations = opt->max_iterations;
    if (strcmp(opt->linesearch, "Backtracking") == 0)
    {
        lbfgs->linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    }
    else if (strcmp(opt->linesearch, "StrongBacktracking") == 0)
    {
        lbfgs->linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    }
    else
    {
        lbfgs->linesearch = LBFGS_LINESEARCH_MORETHUENTE;
    }
    lbfgs->max_linesearch = opt->linesearch_max_iterations;

    /* Set regularization parameters. */
    if (0 < opt->c1)
    {
        lbfgs->orthantwise_c = opt->c1;
        lbfgs->linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    }
    else
    {
        lbfgs->orthantwise_c = 0;
    }
}

static void lbfgs_options_init(lbfgs_training_option_t * result)
{
    result->c1=0;
    result->c2=1.0;
    result->delta=1e-5;
    result->epsilon=1e-5;
    result->linesearch="MoreThuente";
    result->linesearch_max_iterations=20;
    result->max_iterations=INT_MAX;
    result->memory=6;
    result->stop=10;
}

void maxent_attribute_init(maxent_attribute_t* cont)
{
    //memset(cont, 0, sizeof(*cont));
    cont->aid=-1;
    cont->value = 1;
}

void maxent_attribute_set(maxent_attribute_t* cont, int aid, double value)
{
    //crfsuite_attribute_init(cont);
    cont->aid = aid;
    cont->value = value;
}

void maxent_attribute_copy(maxent_attribute_t* dst, const maxent_attribute_t* src)
{
    dst->aid = src->aid;
    dst->value = src->value;
}

void maxent_attribute_swap(maxent_attribute_t* x, maxent_attribute_t* y)
{
    maxent_attribute_t tmp = *x;
    x->aid = y->aid;
    x->value = y->value;
    y->aid = tmp.aid;
    y->value = tmp.value;
}
int maxent_instance_append_attribute(maxent_instance_t* item, const maxent_attribute_t* cont)
{
    if (item->cap_items <= item->num_items)
    {
        item->cap_items = (item->cap_items + 1) * 2;
        item->items = (maxent_attribute_t*)realloc(
                          item->items, sizeof(maxent_attribute_t) * item->cap_items);
    }
    maxent_attribute_copy(&item->items[item->num_items++], cont);
    return 0;
}

void maxent_instance_init(maxent_instance_t* inst)
{
    //memset(inst, 0, sizeof(*inst));
    inst->weight = 1.;
    inst->num_items = 0;
    inst->cap_items = 0;
    inst->items = NULL;
    inst->label = -1;
}
void maxent_instance_set_label(maxent_instance_t* inst, int label)
{
    inst->label=label;
}
void maxent_instance_init_n(maxent_instance_t* inst, int num_items)
{
    //crfsuite_instance_init(inst);
    if(inst->cap_items<num_items&&inst->items!=NULL)
    {
        inst->cap_items=num_items;
        inst->items=(maxent_attribute_t*)realloc(inst->items,inst->cap_items);
    }
    if(inst->items==NULL)
    {
        inst->cap_items = num_items;
        inst->items = (maxent_attribute_t*)calloc(num_items, sizeof(maxent_attribute_t));
    }
    inst->num_items = 0;
    //inst->labels = (int*)calloc(num_items, sizeof(int));
}


void maxent_instance_destroy(maxent_instance_t* inst)
{
    //if(inst->cap_items==0) return;

    if(inst->items!=NULL)
    {
        free(inst->items);
        inst->items=NULL;
    }
    free(inst);
}
void maxent_instance_copy(maxent_instance_t* dst, const maxent_instance_t* src)
{
    int i;

    dst->num_items = src->num_items;
    dst->cap_items = src->cap_items;
    dst->label = src->label;
    dst->items = (maxent_attribute_t*)calloc(dst->num_items, sizeof(maxent_attribute_t));
    dst->weight = src->weight;
    dst->group = src->group;
    for (i = 0; i < dst->num_items; ++i)
    {
        maxent_attribute_copy(&dst->items[i], &src->items[i]);

    }
}
int  maxent_instance_empty(maxent_instance_t* inst)
{
    return (inst->num_items == 0);
}

void maxent_data_init(maxent_data_t* data)
{
    data->num_instances=0;
    data->cap_instances=0;
    data->instances=NULL;
    data->attrs=NULL;
    data->labels=NULL;
}

void maxent_data_init_n(maxent_data_t* data, int n)
{
    maxent_data_init(data);
    data->num_instances = 0;
    data->cap_instances = n;
    data->instances = (maxent_instance_t*)calloc(n, sizeof(maxent_instance_t));
}
void maxent_data_destroy(maxent_data_t* data)
{
    int i;
    maxent_instance_t *inst=NULL;
    maxent_dictionary_destroy(data->attrs);
    maxent_dictionary_destroy(data->labels);
    for (i = 0; i < data->num_instances; ++i)
    {
        inst=&data->instances[i];
        if(inst!=NULL&&inst->items!=NULL)
            free(inst->items);
    }
    free(data->instances);
    free(data);
}
int  maxent_data_append(maxent_data_t* data, const maxent_instance_t* inst)
{
    if (0 < inst->num_items)
    {
        if (data->cap_instances <= data->num_instances)
        {
            data->cap_instances = (data->cap_instances + 1) * 2;
            data->instances = (maxent_instance_t*)realloc(
                                  data->instances, sizeof(maxent_instance_t) * data->cap_instances);
        }
        maxent_instance_copy(&data->instances[data->num_instances++], inst);
    }
    return 0;
}
#define DEFAULT_MAXENT_DATA_SIZE 100
maxent_data_t* maxent_data_create(void)
{
    maxent_data_t* re=NULL;
    re=(maxent_data_t*)malloc(sizeof(maxent_data_t));
    if(re==NULL)return NULL;
    re->num_instances=0;
    re->cap_instances=DEFAULT_MAXENT_DATA_SIZE;
    re->instances = (maxent_instance_t*)calloc(re->cap_instances, sizeof(maxent_instance_t));
    if(re->instances==NULL)
    {
        free(re);
        return NULL;
    }
    re->attrs=maxent_dictionary_create();
    if(re->attrs==NULL)
    {
        free(re->instances);
        free(re);
        return NULL;
    }
    re->labels=maxent_dictionary_create();
    if(re->labels==NULL)
    {
        maxent_dictionary_destroy(re->attrs);
        free(re->instances);
        free(re);
        return NULL;
    }
    return re;


}
lbfgs_training_option_t* lbfgs_training_option_create(void)
{
    lbfgs_training_option_t*  lbfgs_options=NULL;
    lbfgs_options=(lbfgs_training_option_t*)malloc(sizeof(lbfgs_training_option_t));
    if(lbfgs_options==NULL) return NULL;
    lbfgs_options_init(lbfgs_options);
    return lbfgs_options;
}

maxent_model_t* train(maxent_data_t* trainset,double c2)
{
    maxent_model_t* result=NULL;
    int i;
    lbfgs_training_option_t*  lbfgs_options=NULL;
    lbfgs_parameter_t* lbfgsparam=NULL;
    int features,lbret;
    lbfgsfloatval_t *lambda_=NULL;

    result=(maxent_model_t*)malloc(sizeof(maxent_model_t));
    result->instances=trainset->instances;
    result->attrs=trainset->attrs;
    result->labels=trainset->labels;
    result->feats_model=init_feats_model(trainset);
    result->flag=LBFGS_NORMAL;
    features=result->feats_model->num_features;

    //result->lambda=(double*)calloc(features,sizeof(double));
    lambda_=(lbfgsfloatval_t*)calloc(features,sizeof(lbfgsfloatval_t));
    if(lambda_==NULL) return NULL;
    for(i=0; i<features; i++)
        lambda_[i]=0.0;

    lbfgs_options=(lbfgs_training_option_t*)malloc(sizeof(lbfgs_training_option_t));
    lbfgs_options_init(lbfgs_options);
    lbfgsparam=(lbfgs_parameter_t*)malloc(sizeof(lbfgs_parameter_t));
    if(lbfgsparam==NULL)
    {
        free(lambda_);
        return NULL;
    }
    lbfgs_parameter_init(lbfgsparam);
    lbfgs_parameter_update(lbfgsparam,lbfgs_options);

    result->reg2=c2;
    lbret = lbfgs(features,lambda_,NULL,lbfgs_evaluate,lbfgs_progress,result,lbfgsparam);
    result->lbfgs_flag=lbret;
    /**
    if (lbret == LBFGS_CONVERGENCE)
    {
        printf("L-BFGS resulted in convergence\n");
    }
    else if (lbret == LBFGS_STOP)
    {
        printf("L-BFGS terminated with the stopping criteria\n");
    }
    else if (lbret == LBFGSERR_MAXIMUMITERATION)
    {
        printf("L-BFGS terminated with the maximum number of iterations\n");
    }
    else
    {
        printf("L-BFGS terminated with error code (%d)\n", lbret);
    }
    */
    for(i=0; i<features; i++)
    {
        result->feats_model->features[i].weight=lambda_[i];
        //printf("%i %f\n",i,result->feats_model->features[i].weight);
    }
    free(lambda_);
    //free(lbfgs_options);
    free(lbfgsparam);

    return result;
}
maxent_predict_item* predict(maxent_model_t* trainer,maxent_instance_t* instant)
{
    int num_labels_,i,j,fid,dest;
    double lambda_,sum_=0.0;
    maxent_predict_item *re=NULL;
    maxent_attribute_t*_attri=NULL;

    feature_refs_t* attr_feat_ref=NULL,*attr_feat_ref_item=NULL;

    num_labels_=trainer->feats_model->num_labels;
    re=maxent_predict_item_create(num_labels_);
    attr_feat_ref=trainer->feats_model->attributes;
    for(i=0; i<instant->num_items; i++)
    {
        _attri=&instant->items[i];
        if(_attri->aid<0||_attri->aid>=trainer->feats_model->num_attributes) continue;
        attr_feat_ref_item=&attr_feat_ref[_attri->aid];
        for(j=0; j<attr_feat_ref_item->num_features; j++)
        {
            fid=attr_feat_ref_item->fids[j];
            //lambda_=trainer->lambda[fid];
            lambda_=trainer->feats_model->features[fid].weight;
            dest=trainer->feats_model->features[fid].dst;///
            re->items[dest]._value+=lambda_;
        }
    }
    for(i=0; i<num_labels_; i++)
    {
        re->items[i]._value=exp(re->items[i]._value);
        sum_+=re->items[i]._value;
        maxent_dictionary_to_string(trainer->labels,i,&(re->items[i]._label));
    }
    for(i=0; i<num_labels_; i++)
    {
        re->items[i]._value=re->items[i]._value/sum_;
    }
    return re;
}
void maxent_predict_item_destroy(maxent_predict_item* item)
{
    int i;
    if(item->items==NULL)
    {
        free(item);
        return;
    }
    for(i=0; i<item->labels; i++)
    {
        free(item->items[i]._label);
    }
    free(item->items);
    free(item);
}
maxent_predict_item* maxent_predict_item_create(int size_)
{
    maxent_predict_item *re=NULL;
    int i;
    //maxent_predict_item_ *item=NULL;
    re=(maxent_predict_item*)malloc(sizeof(maxent_predict_item));
    re->labels=size_;
    re->items=(maxent_predict_item_*)calloc(re->labels,sizeof(maxent_predict_item_));
    for(i=0; i<re->labels; i++)
    {
        re->items[i]._value=0.0;
        re->items[i]._label=NULL;
    }
    return re;
}
maxent_dictionary_t* maxent_read_dictionary(uint8_t* buffer,uint32_t size_)
{
    cqdb_t *reader=NULL;
    maxent_dictionary_t *result=NULL;

    int number=0,i;
    //char *str=NULL;
    reader= cqdb_reader(buffer,size_);
    number=cqdb_num(reader);
    result=maxent_dictionary_create();
    for(i=0; i<number; i++)
    {
        //str=cqdb_to_string(reader,i);
        maxent_dictionary_get(result,cqdb_to_string(reader,i));
    }
    return result;

}
maxent_model_t* load(char* filename)
{
    maxent_model_t* re=NULL;
    maxent_features_model_t *re_model=NULL;

    FILE *fp = NULL;
    uint8_t* p = NULL,*buffer=NULL,*buffer_=NULL;
    uint32_t size_=0;
    header_t *header = NULL;

    fp = fopen(filename, "rb");
    if (fp == NULL) return NULL;

    fseek(fp, 0, SEEK_END);

    size_ = (uint32_t)ftell(fp);///文件的大小
    ///ftell:       得到流式文件的当前读写位置,其返回值是当前读写位置偏离文件头部的字节数
    fseek(fp, 0, SEEK_SET);

    buffer =buffer_ = (uint8_t*)malloc(size_ + 16);
    re=(maxent_model_t*)malloc(sizeof(maxent_model_t));
    re_model=(maxent_features_model_t*)malloc(sizeof(maxent_features_model_t));
    if(re==NULL||re_model==NULL)
    {
        if(re!=NULL) free(re);
        if(re_model!=NULL) free(re_model);
        return NULL;
    };
    re->attrs=NULL;
    re->instances=NULL;
    re->labels=NULL;
    re->feats_model=NULL;
    re_model->attributes=NULL;
    re_model->features=NULL;

    re->feats_model=re_model;
    while ((uintptr_t)buffer % 16 != 0)  ///内存对齐
    {
        ++buffer;
    }
    if (fread(buffer, 1, size_, fp) != size_)
    {
        free(buffer_);
        free(re->feats_model);
        free(re);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    header = (header_t*)calloc(1, sizeof(header_t));
    p = buffer;
    p += read_uint8_array(p, header->magic, sizeof(header->magic));
    p += read_uint32(p, &header->size);
    p += read_uint32(p, &header->version);
    p += read_uint32(p, &header->num_features);
    p += read_uint32(p, &header->num_labels);
    p += read_uint32(p, &header->num_attributes);
    p += read_uint32(p, &header->num_instances);
    p += read_uint8(p, &header->flag);
    p += read_uint32(p, &header->off_options);
    p += read_uint32(p, &header->off_features);
    p += read_uint32(p, &header->off_labels);
    p += read_uint32(p, &header->off_attrs);
    p += read_uint32(p, &header->off_attrrefs);


    maxent_model_init_header(re,header);
    p=buffer+header->off_options;
    p+=read_double(p,&re->reg2);
    p+=read_uint32(p,(uint32_t*)&re->lbfgs_flag);

    re->feats_model->features=maxent_read_features(buffer,header);
    re->feats_model->attributes=maxent_read_feature_ref(buffer,header,re->feats_model->num_attributes);
    re->labels = maxent_read_dictionary(buffer + header->off_labels,header->off_attrs-header->num_labels);
    re->attrs = maxent_read_dictionary(buffer + header->off_attrs,header->off_attrrefs-header->off_attrs);
    free(buffer_);
    return re;
}



void maxent_dictionary_destroy(maxent_dictionary_t* dic)
{
    quark_delete(dic);
    //free(dic);
}

int maxent_dictionary_get(maxent_dictionary_t* dic, const char *str)
{
    return quark_get(dic, str);
}
int maxent_dictionary_to_id(maxent_dictionary_t* dic, const char *str)
{
    return quark_to_id(dic, str);
}

int maxent_dictionary_to_string(maxent_dictionary_t* dic, int id, char **pstr)
{
    const char *str = quark_to_string(dic, id);
    if (str != NULL)
    {
        char *dst = (char*)malloc(strlen(str)+1);
        if (dst)
        {
            strcpy(dst, str);
            *pstr = dst;
            return 0;
        }
    }
    return 1;
}

int maxent_dictionary_num(maxent_dictionary_t* dic)
{
    return quark_num(dic);
}

void maxent_dictionary_free(maxent_dictionary_t* dic, const char *str)
{
    free((char*)str);
}

maxent_dictionary_t* maxent_dictionary_create(void)
{
    maxent_dictionary_t* result=NULL;
    result=quark_new();
    return result;
}
