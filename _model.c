#include <Python.h>
#include "maxent.h"
#define DEFAULT_MAXENT_MODEL_NAME "maxent_model_zeng"
#define EXTRACT_STR_FROM_OBJ(str,item)							\
if(item&&PyUnicode_Check(item)) str=PyString_AsString(PyUnicode_AsUTF8String(item));	\
else if(item&&PyString_Check(item))str=PyString_AsString(item);	


typedef struct
{
    PyObject_HEAD
    maxent_data_t*      _data;
    maxent_model_t*     _model;
} MaxEntTrainer;

static PyObject* MaxEntTrainer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    MaxEntTrainer *result=NULL;
    result = (MaxEntTrainer *)type->tp_alloc(type, 0);
    result->_model=NULL;
    result->_data=NULL;
    return (PyObject *)result;
}
///析构:
static int MaxEntTrainer_clear(MaxEntTrainer *self)
{
    if(self->_model!=NULL&&self->_data!=NULL)
    {
        
        if(self->_model->attrs==self->_data->attrs)
        {
            maxent_model_destroy_(self->_model);
            maxent_data_destroy(self->_data);
        }
        else
        {
            maxent_model_destroy(self->_model);
            maxent_data_destroy(self->_data);
        }
    }
    else if(self->_model!=NULL)
    {
        maxent_model_destroy(self->_model);
        self->_model=NULL;
    }
    else if(self->_data!=NULL)
    {
        maxent_data_destroy(self->_data);
        self->_data=NULL;
    }
    Py_XDECREF(self);
    return 0;
}
static void MaxEntTrainer_dealloc(MaxEntTrainer* self)
{
    MaxEntTrainer_clear(self);
#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif
}
///添加数据:单条数据 ['a','b','c'] 'label'
///Python:['a','b','c'], 'label'

static PyObject* MaxEntTrainer_append_instance(MaxEntTrainer *self, PyObject *args)
{
    PyObject  *items,*item=NULL,*labelItem=NULL;
    char *label=NULL,*itemStr=NULL;
    //const char* itemStr_;
    int itemsLen,i;
    maxent_attribute_t*     _attr=NULL;
    maxent_instance_t*      _inst=NULL;

    if (!PyArg_ParseTuple(args, "OO",&items,&labelItem))
        return NULL;
    ///判断非list退出
EXTRACT_STR_FROM_OBJ(label,labelItem);
    if(!PyList_Check(items))
    {
        PyErr_SetString(PyExc_TypeError,"typeError:must be List.");
        return NULL;
    }

    if(self->_data==NULL)
    {
        self->_data=maxent_data_create();
        if(self->_data==NULL)
            return PyErr_NoMemory();
    }
    itemsLen= (int)PyList_Size(items);
    if(itemsLen<1)
    {
        PyErr_SetString(PyExc_RuntimeError,"the length of List is too less.");
        return NULL;
    }
    _attr=(maxent_attribute_t*)malloc(sizeof(maxent_attribute_t));
    if(_attr==NULL)
        return PyErr_NoMemory();
    _inst=(maxent_instance_t*)malloc(sizeof(maxent_instance_t));

    if(_inst==NULL)
    {
        free(_attr);
        return PyErr_NoMemory();
    }
    maxent_instance_init(_inst);
    maxent_instance_init_n(_inst,itemsLen);

/////////////////////////////////////////////////////////////////////////////
    for(i=0; i<itemsLen; i++)
    {
        item=PyList_GetItem(items,i);
        //if(!PyUnicode_Check(item))continue;

#if PY_MAJOR_VERSION >= 3
        itemStr=PyUnicode_AsUTF8(item);
#else
EXTRACT_STR_FROM_OBJ(itemStr,item);

#endif
        //itemStr=PyUnicode_AsUTF8(item);

        maxent_attribute_init(_attr);
        maxent_attribute_set(_attr,maxent_dictionary_get(self->_data->attrs,itemStr),1.0);
        maxent_instance_append_attribute(_inst,_attr);
    }
    maxent_instance_set_label(_inst,maxent_dictionary_get(self->_data->labels,label));
    maxent_data_append(self->_data,_inst);
    ///free
    free(_attr);
    if(_inst->items!=NULL)free(_inst->items);
    free(_inst);
    Py_RETURN_NONE;
}
///训练模型
///     异常:model中的数据量<data中的数据量
///Python:可选 double c2
static PyObject* MaxEntTrainer_train(MaxEntTrainer *self, PyObject *args)
{
    double c2_=1.0;
    //int i;

    if (!PyArg_ParseTuple(args, "|d",&c2_))
        return NULL;

    if(self->_data==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"NoDataError.");
        return NULL;
    }
    ///异常model非空  装载模型-添加数据-训练模型 添加数据-训练模型-添加数据-训练模型
    if(self->_model!=NULL)
    {
        if(self->_model->attrs==self->_data->attrs)
            maxent_model_destroy_(self->_model);
        else
            maxent_model_destroy(self->_model);
    }
    self->_model=train(self->_data,c2_);
    if(self->_model==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"TrainError.");
        return NULL;
    }
    return Py_BuildValue("i",self->_model->lbfgs_flag);
}
///保存模型
///Python:'filename'
static PyObject* MaxEntTrainer_save_model(MaxEntTrainer *self, PyObject *args)
{
PyObject  *item=NULL;
    char* fileName=DEFAULT_MAXENT_MODEL_NAME;
    int ret;
    if(self->_model==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"NoModelError.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "|O",&item))
        return NULL;
EXTRACT_STR_FROM_OBJ(fileName,item);
    ret=save(self->_model,fileName);
    if(ret==-1) Py_RETURN_FALSE;
    else Py_RETURN_TRUE;
}
///模型预测
///Pytholn :['a','b','c'], 返回值为Python dict字典对象
static PyObject* MaxEntTrainer_predict(MaxEntTrainer *self, PyObject *args)
{
    PyObject  *items,*dictItem,*item=NULL;
    char *itemStr=NULL;
    int i,itemsLen;
    maxent_attribute_t*     _attr=NULL;
    maxent_instance_t*      _inst=NULL;
    maxent_predict_item*    _ret=NULL;

    if(self->_model==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"NoModelError.");
        return NULL;
    }
    if (!PyArg_ParseTuple(args, "O",&items))
        return NULL;
    ///判断非list退出
    if(!PyList_Check(items))
    {
        PyErr_SetString(PyExc_TypeError,"typeError:must be List.");
        return NULL;
    }

    itemsLen= (int)PyList_Size(items);
    if(itemsLen<1)
    {
        PyErr_SetString(PyExc_RuntimeError,"the length of List is too less.");
        return NULL;
    }
    _attr=(maxent_attribute_t*)malloc(sizeof(maxent_attribute_t));
    if(_attr==NULL)
        return PyErr_NoMemory();
    _inst=(maxent_instance_t*)malloc(sizeof(maxent_instance_t));

    if(_inst==NULL)
    {
        free(_attr);
        return PyErr_NoMemory();
    }
    maxent_instance_init(_inst);
    maxent_instance_init_n(_inst,itemsLen);


/////////////////////////////////////////////////////////////////////////////
    for(i=0; i<itemsLen; i++)
    {
        item=PyList_GetItem(items,i);
        //if(!PyUnicode_Check(item))continue;


#if PY_MAJOR_VERSION >= 3
        itemStr=PyUnicode_AsUTF8(item);
#else
EXTRACT_STR_FROM_OBJ(itemStr,item);
       
#endif


        maxent_attribute_init(_attr);
        maxent_attribute_set(_attr,maxent_dictionary_to_id(self->_model->attrs,itemStr),1.0);

        maxent_instance_append_attribute(_inst,_attr);
    }
    ///========================预测====================================
    _ret=predict(self->_model,_inst);
    if(_ret==NULL)
    {
        free(_attr);
        if(_inst->items!=NULL)free(_inst->items);
        free(_inst);
        PyErr_SetString(PyExc_RuntimeError,"PredictError.");
        return NULL;
    }
    dictItem=PyDict_New();
    for(i=0; i<_ret->labels; i++)
    {
        PyDict_SetItemString(dictItem,_ret->items[i]._label,Py_BuildValue("d",_ret->items[i]._value));
    }

    ///========================预测====================================

    ///free
    free(_attr);
    if(_inst->items!=NULL)free(_inst->items);
    free(_inst);
    maxent_predict_item_destroy(_ret);
    return dictItem;
}
///装载模型
///Python:'filename' 返回值为True False
static PyObject* MaxEntTrainer_load_model(MaxEntTrainer *self, PyObject *args)
{
PyObject  *item=NULL;
    char* fileName=DEFAULT_MAXENT_MODEL_NAME;
    if (!PyArg_ParseTuple(args, "O",&item))
        return NULL;
EXTRACT_STR_FROM_OBJ(fileName,item);
    self->_model=load(fileName);
    if(self->_model==NULL) Py_RETURN_FALSE;
    else Py_RETURN_TRUE;
}
static PyObject* MaxEntTrainer_data_size(MaxEntTrainer *self)
{
    if(self->_data!=NULL)
        return  Py_BuildValue("i",self->_data->num_instances);
    else
        Py_RETURN_NONE;
}
static PyObject* MaxEntTrainer_data_exist(MaxEntTrainer *self)
{
    if(self->_data==NULL) Py_RETURN_FALSE;
    else Py_RETURN_TRUE;
}
static PyObject* MaxEntTrainer_model_exist(MaxEntTrainer *self)
{
    if(self->_model==NULL) Py_RETURN_FALSE;
    else Py_RETURN_TRUE;
}
static PyMethodDef MaxEntTrainer_methods[] =
{
    {"append", (PyCFunction)MaxEntTrainer_append_instance, METH_VARARGS,PyDoc_STR("MaxEnt:append data to Trainer")},
    {"train", (PyCFunction)MaxEntTrainer_train, METH_VARARGS,PyDoc_STR("MaxEnt:train model")},
    {"save", (PyCFunction)MaxEntTrainer_save_model, METH_VARARGS,PyDoc_STR("MaxEnt:save model to file")},
    {"load", (PyCFunction)MaxEntTrainer_load_model, METH_VARARGS,PyDoc_STR("MaxEnt:load model from file")},
    {"predict", (PyCFunction)MaxEntTrainer_predict, METH_VARARGS,PyDoc_STR("MaxEnt:predict the labels")},
    {"size", (PyCFunction)MaxEntTrainer_data_size, METH_NOARGS,PyDoc_STR("MaxEnt:get the length of data")},
    {"exist_data", (PyCFunction)MaxEntTrainer_data_exist, METH_NOARGS,PyDoc_STR("MaxEnt:check data exist")},
    {"exist_model", (PyCFunction)MaxEntTrainer_model_exist, METH_NOARGS,PyDoc_STR("MaxEnt:check model exist")},
    {NULL}
};

static PyObject * _model_version(PyObject *self)
{
    return Py_BuildValue("s","0.1.0");
}
static PyObject * _model_author(PyObject *self)
{
    return Py_BuildValue("s","xh.along(zengxiaolong2015@163.com)");
}

static PyMethodDef _model_methods[] =
{
    {"version", (PyCFunction)_model_version, METH_NOARGS,PyDoc_STR("version")},
    {"author", (PyCFunction)_model_author, METH_NOARGS,PyDoc_STR("author")},
    {NULL}
};
static PyTypeObject MaxEntTrainer_tp =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    //PyObject_HEAD_INIT(NULL)
    "MaxEntTrainer",         /* tp_name */
    sizeof(MaxEntTrainer),          /* tp_basicsize */
    0,                       /* tp_itemsize */
    (destructor)MaxEntTrainer_dealloc,/* tp_dealloc */
    0,                       /* tp_print */
    0,                       /* tp_getattr */
    0,                       /* tp_setattr */
    0,                       /* tp_reserved */
    0,                       /* tp_repr */
    0,                       /* tp_as_number */
    0,                       /* tp_as_sequence */
    0,                       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    0,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE, /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    (inquiry)MaxEntTrainer_clear,  /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    MaxEntTrainer_methods,                       /* tp_methods */
    0,                       /* tp_members */
    0,                       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    0,                       /* tp_init */
    0,                       /* tp_alloc */
    MaxEntTrainer_new,             /* tp_new */
};
#define _model__doc__ "entroPy.model"
#if PY_MAJOR_VERSION >= 3
static PyModuleDef _modelmodule =
{
    PyModuleDef_HEAD_INIT,
    "_model",
    _model__doc__,
    -1,
    //NULL
    _model_methods
};
#define INITERROR return NULL
PyMODINIT_FUNC PyInit__model(void)
#else
#define INITERROR return
PyMODINIT_FUNC init_model(void)
#endif
{
    PyObject *_model_mod;
#if PY_MAJOR_VERSION >= 3
    _model_mod = PyModule_Create(&_modelmodule);
    if (_model_mod == NULL)
#else
    if (!(_model_mod = Py_InitModule3("_model", _model_methods, _model__doc__)))
#endif
        INITERROR;
    

    MaxEntTrainer_tp.tp_alloc = PyType_GenericAlloc;
    MaxEntTrainer_tp.tp_free = PyObject_Del;
    if (PyType_Ready(&MaxEntTrainer_tp) < 0)
        INITERROR;
    if (PyType_Ready(&MaxEntTrainer_tp) < 0)
        INITERROR;
    Py_INCREF(&MaxEntTrainer_tp);

    if(PyModule_AddObject(_model_mod, "MaxEntTrainer", (PyObject *) &MaxEntTrainer_tp)<0)
        INITERROR;
#if PY_MAJOR_VERSION >= 3
    return _model_mod;
#endif
}
