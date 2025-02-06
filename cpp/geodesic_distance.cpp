#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "geodesic_distance_2d.h"
#include "geodesic_distance_3d.h"

static PyObject* geodesic2d_fast_marching_wrapper(PyObject* self, PyObject* args) {
    PyObject *I_obj, *S_obj;
    PyArrayObject *arr_I, *arr_Seed, *distance;
    npy_intp *shape;

    if (!PyArg_ParseTuple(args, "OO", &I_obj, &S_obj))
        return NULL;

    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr_I == NULL) return NULL;

    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(S_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (arr_Seed == NULL) {
        Py_DECREF(arr_I);
        return NULL;
    }

    shape = PyArray_DIMS(arr_I);
    npy_intp dims[2] = {shape[0], shape[1]};
    distance = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (distance == NULL) {
        Py_DECREF(arr_I);
        Py_DECREF(arr_Seed);
        return NULL;
    }

    geodesic2d_fast_marching(
        (const float*)PyArray_DATA(arr_I),
        (const unsigned char*)PyArray_DATA(arr_Seed),
        (float*)PyArray_DATA(distance),
        shape[0], shape[1]
    );

    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    return (PyObject*)distance;
}

static PyObject* geodesic2d_raster_scan_wrapper(PyObject* self, PyObject* args) {
    PyObject *I_obj, *S_obj;
    PyArrayObject *arr_I, *arr_Seed, *distance;
    npy_intp *shape;

    if (!PyArg_ParseTuple(args, "OO", &I_obj, &S_obj))
        return NULL;

    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr_I == NULL) return NULL;

    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(S_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (arr_Seed == NULL) {
        Py_DECREF(arr_I);
        return NULL;
    }

    shape = PyArray_DIMS(arr_I);
    npy_intp dims[2] = {shape[0], shape[1]};
    distance = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (distance == NULL) {
        Py_DECREF(arr_I);
        Py_DECREF(arr_Seed);
        return NULL;
    }

    geodesic2d_raster_scan(
        (const float*)PyArray_DATA(arr_I),
        (const unsigned char*)PyArray_DATA(arr_Seed),
        (float*)PyArray_DATA(distance),
        shape[0], shape[1]
    );

    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    return (PyObject*)distance;
}

static PyObject* geodesic3d_fast_marching_wrapper(PyObject* self, PyObject* args) {
    PyObject *I_obj, *S_obj;
    PyArrayObject *arr_I, *arr_Seed, *distance;
    npy_intp *shape;

    if (!PyArg_ParseTuple(args, "OO", &I_obj, &S_obj))
        return NULL;

    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr_I == NULL) return NULL;

    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(S_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (arr_Seed == NULL) {
        Py_DECREF(arr_I);
        return NULL;
    }

    shape = PyArray_DIMS(arr_I);
    npy_intp dims[3] = {shape[0], shape[1], shape[2]};
    distance = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (distance == NULL) {
        Py_DECREF(arr_I);
        Py_DECREF(arr_Seed);
        return NULL;
    }

    geodesic3d_fast_marching(
        (const float*)PyArray_DATA(arr_I),
        (const unsigned char*)PyArray_DATA(arr_Seed),
        (float*)PyArray_DATA(distance),
        shape[0], shape[1], shape[2]
    );

    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    return (PyObject*)distance;
}

static PyObject* geodesic3d_raster_scan_wrapper(PyObject* self, PyObject* args) {
    PyObject *I_obj, *S_obj;
    float lambda;
    int iteration;
    PyArrayObject *arr_I, *arr_Seed, *distance;
    npy_intp *shape;

    if (!PyArg_ParseTuple(args, "OOfi", &I_obj, &S_obj, &lambda, &iteration))
        return NULL;

    arr_I = (PyArrayObject*)PyArray_FROM_OTF(I_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr_I == NULL) return NULL;

    arr_Seed = (PyArrayObject*)PyArray_FROM_OTF(S_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (arr_Seed == NULL) {
        Py_DECREF(arr_I);
        return NULL;
    }

    shape = PyArray_DIMS(arr_I);
    npy_intp dims[3] = {shape[0], shape[1], shape[2]};
    distance = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (distance == NULL) {
        Py_DECREF(arr_I);
        Py_DECREF(arr_Seed);
        return NULL;
    }

    geodesic3d_raster_scan(
        (const float*)PyArray_DATA(arr_I),
        (const unsigned char*)PyArray_DATA(arr_Seed),
        (float*)PyArray_DATA(distance),
        shape[0], shape[1], shape[2],
        lambda, iteration
    );

    Py_DECREF(arr_I);
    Py_DECREF(arr_Seed);
    return (PyObject*)distance;
}

static PyMethodDef GeodesicMethods[] = {
    {"geodesic2d_fast_marching", geodesic2d_fast_marching_wrapper, METH_VARARGS,
     "Compute geodesic distance using fast marching method (2D)"},
    {"geodesic2d_raster_scan", geodesic2d_raster_scan_wrapper, METH_VARARGS,
     "Compute geodesic distance using raster scan method (2D)"},
    {"geodesic3d_fast_marching", geodesic3d_fast_marching_wrapper, METH_VARARGS,
     "Compute geodesic distance using fast marching method (3D)"},
    {"geodesic3d_raster_scan", geodesic3d_raster_scan_wrapper, METH_VARARGS,
     "Compute geodesic distance using raster scan method (3D)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef geodesic_module = {
    PyModuleDef_HEAD_INIT,
    "geodesic_distance",
    NULL,
    -1,
    GeodesicMethods
};

PyMODINIT_FUNC PyInit_geodesic_distance(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&geodesic_module);
}