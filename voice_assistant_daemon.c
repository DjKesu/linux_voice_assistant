#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <syslog.h>
#include <Python.h>

static void skeleton_daemon()
{
    pid_t pid;

    pid = fork();
    if (pid < 0) exit(EXIT_FAILURE);
    if (pid > 0) exit(EXIT_SUCCESS);

    if (setsid() < 0) exit(EXIT_FAILURE);

    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    pid = fork();
    if (pid < 0) exit(EXIT_FAILURE);
    if (pid > 0) exit(EXIT_SUCCESS);

    umask(0);
    chdir("/");

    for (int x = sysconf(_SC_OPEN_MAX); x >= 0; x--)
    {
        close(x);
    }

    openlog("linuxvoiceassistant", LOG_PID, LOG_DAEMON);
}

int main()
{
    skeleton_daemon();
    syslog(LOG_NOTICE, "Linux Voice Assistant daemon started.");

    Py_Initialize();
    syslog(LOG_NOTICE, "Python interpreter initialized.");

    PyObject *pName, *pModule, *pFunc, *pValue;

    pName = PyUnicode_DecodeFSDefault("wake_word_detection_lib");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        syslog(LOG_NOTICE, "Wake word detection module loaded successfully.");
        pFunc = PyObject_GetAttrString(pModule, "wake_word_detected");

        if (pFunc && PyCallable_Check(pFunc)) {
            syslog(LOG_NOTICE, "Wake word detection function found and is callable.");
            while (1) {
                syslog(LOG_NOTICE, "Listening for wake word...");
                pValue = PyObject_CallObject(pFunc, NULL);
                if (pValue != NULL) {
                    if (PyLong_AsLong(pValue) == 1) {
                        syslog(LOG_NOTICE, "Wake word detected! Ready for command.");
                        // Future: Add code here to listen for and process the command
                    }
                    Py_DECREF(pValue);
                } else {
                    syslog(LOG_ERR, "Wake word detection function call failed.");
                    PyErr_Print();
                }
            }
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            syslog(LOG_ERR, "Cannot find function 'wake_word_detected'");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        syslog(LOG_ERR, "Failed to load 'wake_word_detection_lib'");
    }

    syslog(LOG_NOTICE, "Linux Voice Assistant daemon shutting down.");
    Py_Finalize();
    closelog();

    return EXIT_SUCCESS;
}