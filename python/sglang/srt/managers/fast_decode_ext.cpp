// Fast C++ extension for decode batch result processing.
// Avoids per-step Python attribute lookups by caching request constants.

#include <Python.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

struct CachedReqInfo {
    int64_t max_new_tokens;
    int64_t vocab_size;
    int64_t nan_replacement_token;  // token to replace NaN with
    std::unordered_set<int64_t> stop_token_ids;  // from sampling_params.stop_token_ids
    std::unordered_set<int64_t> eos_token_ids;   // from req.eos_token_ids
    int64_t primary_eos_token_id;  // from tokenizer.eos_token_id (-1 if none)
    std::unordered_set<int64_t> additional_stop_ids;  // from tokenizer.additional_stop_token_ids
    bool ignore_eos;
    bool has_stop_strs;
};

static std::unordered_map<uintptr_t, CachedReqInfo> g_req_cache;

static void extract_int_set(PyObject* obj, std::unordered_set<int64_t>& out) {
    if (obj == nullptr || obj == Py_None) return;
    PyObject* iter = PyObject_GetIter(obj);
    if (!iter) { PyErr_Clear(); return; }
    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        if (PyLong_Check(item)) {
            out.insert(PyLong_AsLongLong(item));
        }
        Py_DECREF(item);
    }
    Py_DECREF(iter);
}

static int64_t get_first_from_py_set(PyObject* obj) {
    if (obj == nullptr || obj == Py_None) return -1;
    PyObject* iter = PyObject_GetIter(obj);
    if (!iter) { PyErr_Clear(); return -1; }
    PyObject* item = PyIter_Next(iter);
    int64_t val = -1;
    if (item) {
        if (PyLong_Check(item)) val = PyLong_AsLongLong(item);
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    return val;
}

static PyObject* cache_request(PyObject* self, PyObject* args) {
    PyObject* req;
    if (!PyArg_ParseTuple(args, "O", &req)) return nullptr;

    uintptr_t key = (uintptr_t)req;
    CachedReqInfo info;
    info.primary_eos_token_id = -1;
    info.nan_replacement_token = -1;

    // sampling_params
    PyObject* sp = PyObject_GetAttrString(req, "sampling_params");
    if (!sp) return nullptr;

    PyObject* mnt = PyObject_GetAttrString(sp, "max_new_tokens");
    info.max_new_tokens = PyLong_AsLongLong(mnt);
    Py_DECREF(mnt);

    PyObject* ie = PyObject_GetAttrString(sp, "ignore_eos");
    info.ignore_eos = (ie == Py_True);
    Py_DECREF(ie);

    PyObject* stids = PyObject_GetAttrString(sp, "stop_token_ids");
    extract_int_set(stids, info.stop_token_ids);
    Py_DECREF(stids);

    PyObject* sstrs = PyObject_GetAttrString(sp, "stop_strs");
    info.has_stop_strs = (sstrs != Py_None && PyObject_IsTrue(sstrs));
    Py_DECREF(sstrs);

    PyObject* sregex = PyObject_GetAttrString(sp, "stop_regex_strs");
    if (sregex != Py_None && PyObject_IsTrue(sregex)) info.has_stop_strs = true;
    Py_DECREF(sregex);
    Py_DECREF(sp);

    // vocab_size
    PyObject* vs = PyObject_GetAttrString(req, "vocab_size");
    info.vocab_size = PyLong_AsLongLong(vs);
    Py_DECREF(vs);

    // eos_token_ids
    PyObject* eids = PyObject_GetAttrString(req, "eos_token_ids");
    extract_int_set(eids, info.eos_token_ids);
    // NaN replacement: prefer eos_token_ids, then stop_token_ids
    if (!info.stop_token_ids.empty())
        info.nan_replacement_token = *info.stop_token_ids.begin();
    if (!info.eos_token_ids.empty())
        info.nan_replacement_token = *info.eos_token_ids.begin();
    Py_DECREF(eids);

    // tokenizer
    PyObject* tok = PyObject_GetAttrString(req, "tokenizer");
    if (tok && tok != Py_None) {
        PyObject* eos_id = PyObject_GetAttrString(tok, "eos_token_id");
        if (eos_id && eos_id != Py_None && PyLong_Check(eos_id)) {
            info.primary_eos_token_id = PyLong_AsLongLong(eos_id);
        }
        Py_XDECREF(eos_id);

        PyObject* addl = PyObject_GetAttrString(tok, "additional_stop_token_ids");
        if (addl && addl != Py_None) {
            extract_int_set(addl, info.additional_stop_ids);
        }
        Py_XDECREF(addl);
    }
    Py_XDECREF(tok);

    g_req_cache[key] = std::move(info);
    Py_RETURN_NONE;
}

static PyObject* uncache_request(PyObject* self, PyObject* args) {
    PyObject* req;
    if (!PyArg_ParseTuple(args, "O", &req)) return nullptr;
    g_req_cache.erase((uintptr_t)req);
    Py_RETURN_NONE;
}

static PyObject* clear_cache(PyObject* self, PyObject* args) {
    g_req_cache.clear();
    Py_RETURN_NONE;
}

static PyObject* g_finish_length_cls = nullptr;
static PyObject* g_finish_matched_token_cls = nullptr;
static PyObject* g_finish_matched_str_cls = nullptr;

static PyObject* set_finish_classes(PyObject* self, PyObject* args) {
    PyObject *fl, *fmt, *fms;
    if (!PyArg_ParseTuple(args, "OOO", &fl, &fmt, &fms)) return nullptr;
    Py_XDECREF(g_finish_length_cls);
    Py_XDECREF(g_finish_matched_token_cls);
    Py_XDECREF(g_finish_matched_str_cls);
    g_finish_length_cls = fl; Py_INCREF(fl);
    g_finish_matched_token_cls = fmt; Py_INCREF(fmt);
    g_finish_matched_str_cls = fms; Py_INCREF(fms);
    Py_RETURN_NONE;
}

// Returns: list of indices of requests that finished in this step.
// Handles: token append, max_new_tokens, EOS/stop tokens, vocab boundary,
// to_finish abort, and string-based finish (Python fallback).
static PyObject* fast_decode_step(PyObject* self, PyObject* args) {
    PyObject* reqs_list;
    PyObject* next_ids_list;
    int enable_overlap;

    if (!PyArg_ParseTuple(args, "OOp", &reqs_list, &next_ids_list, &enable_overlap))
        return nullptr;

    Py_ssize_t num_reqs = PyList_GET_SIZE(reqs_list);
    PyObject* finished_indices = PyList_New(0);

    for (Py_ssize_t i = 0; i < num_reqs; i++) {
        PyObject* req = PyList_GET_ITEM(reqs_list, i);

        // Skip already-finished requests in overlap mode
        if (enable_overlap) {
            PyObject* fr = PyObject_GetAttrString(req, "finished_reason");
            bool already_done = (fr != Py_None);
            Py_DECREF(fr);
            if (already_done) continue;
        }

        // Append next_token_id to output_ids
        PyObject* next_tid = PyList_GET_ITEM(next_ids_list, i);
        int64_t token_id = PyLong_AsLongLong(next_tid);

        PyObject* output_ids = PyObject_GetAttrString(req, "output_ids");
        PyList_Append(output_ids, next_tid);
        Py_ssize_t output_len = PyList_GET_SIZE(output_ids);

        // --- Fast finish checking using cached info ---
        uintptr_t key = (uintptr_t)req;
        auto it = g_req_cache.find(key);

        if (it != g_req_cache.end()) {
            const CachedReqInfo& info = it->second;

            // Check: to_finish (abort)
            PyObject* to_finish = PyObject_GetAttrString(req, "to_finish");
            if (to_finish != Py_None && PyObject_IsTrue(to_finish)) {
                PyObject_SetAttrString(req, "finished_reason", to_finish);
                PyObject_SetAttrString(req, "to_finish", Py_None);
                Py_DECREF(to_finish);
                Py_DECREF(output_ids);
                goto handle_finished;
            }
            Py_DECREF(to_finish);

            // Check: max_new_tokens
            if (output_len >= info.max_new_tokens) {
                if (g_finish_length_cls) {
                    PyObject* arg = PyLong_FromLongLong(info.max_new_tokens);
                    PyObject* reason = PyObject_CallFunctionObjArgs(g_finish_length_cls, arg, nullptr);
                    PyObject_SetAttrString(req, "finished_reason", reason);
                    Py_DECREF(reason);
                    Py_DECREF(arg);
                }
                PyObject* fl = PyLong_FromLongLong(info.max_new_tokens);
                PyObject_SetAttrString(req, "finished_len", fl);
                Py_DECREF(fl);
                Py_DECREF(output_ids);
                goto handle_finished;
            }

            // Check: vocab boundary (NaN detection) — always checked
            if (token_id > info.vocab_size || token_id < 0) {
                // Replace the bad token
                Py_ssize_t offset = output_len - 1;
                if (info.nan_replacement_token >= 0) {
                    PyObject* replacement = PyLong_FromLongLong(info.nan_replacement_token);
                    PyList_SetItem(output_ids, offset, replacement);
                    // PyList_SetItem steals the reference
                }
                if (g_finish_matched_str_cls) {
                    PyObject* matched = PyUnicode_FromString("NaN happened");
                    PyObject* reason = PyObject_CallFunctionObjArgs(
                        g_finish_matched_str_cls, matched, nullptr);
                    PyObject_SetAttrString(req, "finished_reason", reason);
                    Py_DECREF(reason);
                    Py_DECREF(matched);
                }
                PyObject* fl = PyLong_FromSsize_t(offset + 1);
                PyObject_SetAttrString(req, "finished_len", fl);
                Py_DECREF(fl);
                Py_DECREF(output_ids);
                goto handle_finished;
            }

            // Check: EOS/stop tokens (gated by ignore_eos)
            if (!info.ignore_eos) {
                bool matched = false;

                if (!info.stop_token_ids.empty() && info.stop_token_ids.count(token_id))
                    matched = true;
                else if (!info.eos_token_ids.empty() && info.eos_token_ids.count(token_id))
                    matched = true;
                else if (info.primary_eos_token_id >= 0 && token_id == info.primary_eos_token_id)
                    matched = true;
                else if (!info.additional_stop_ids.empty() && info.additional_stop_ids.count(token_id))
                    matched = true;

                if (matched) {
                    if (g_finish_matched_token_cls) {
                        PyObject* arg = PyLong_FromLongLong(token_id);
                        PyObject* reason = PyObject_CallFunctionObjArgs(
                            g_finish_matched_token_cls, arg, nullptr);
                        PyObject_SetAttrString(req, "finished_reason", reason);
                        Py_DECREF(reason);
                        Py_DECREF(arg);
                    }
                    PyObject* fl = PyLong_FromSsize_t(output_len);
                    PyObject_SetAttrString(req, "finished_len", fl);
                    Py_DECREF(fl);
                    Py_DECREF(output_ids);
                    goto handle_finished;
                }
            }

            // Check: stop strings (Python fallback)
            if (info.has_stop_strs) {
                PyObject* result = PyObject_CallMethod(req,
                    "_check_str_based_finish", nullptr);
                bool str_finished = (result && PyObject_IsTrue(result));
                Py_XDECREF(result);
                if (str_finished) {
                    Py_DECREF(output_ids);
                    goto handle_finished;
                }
            }

            Py_DECREF(output_ids);
            continue;  // Not finished

        } else {
            Py_DECREF(output_ids);
            // No cached info — fall back to Python check_finished
            PyObject* result = PyObject_CallMethod(req, "check_finished", nullptr);
            Py_XDECREF(result);

            PyObject* fr = PyObject_GetAttrString(req, "finished_reason");
            bool is_finished = (fr != Py_None);
            Py_DECREF(fr);
            if (!is_finished) continue;
        }

handle_finished:
        {
            PyObject* idx = PyLong_FromSsize_t(i);
            PyList_Append(finished_indices, idx);
            Py_DECREF(idx);
        }
    }

    return finished_indices;
}

static PyMethodDef methods[] = {
    {"cache_request", cache_request, METH_VARARGS,
     "Cache a request's constants for fast decode checking."},
    {"uncache_request", uncache_request, METH_VARARGS,
     "Remove a request from the cache."},
    {"clear_cache", clear_cache, METH_NOARGS,
     "Clear the entire request cache."},
    {"set_finish_classes", set_finish_classes, METH_VARARGS,
     "Set the FINISH_LENGTH, FINISH_MATCHED_TOKEN, FINISH_MATCHED_STR classes."},
    {"fast_decode_step", fast_decode_step, METH_VARARGS,
     "Fast decode batch result processing. Returns list of finished indices."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_fast_decode_ext",
    "Fast C++ extension for decode batch result processing",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__fast_decode_ext(void) {
    return PyModule_Create(&module_def);
}
