import os
import sys
import json
import time
import shutil
import signal
import logging
from functools import partial

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(filename)s[%(lineno)d]: %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
wlog = logger.info


def auto_select_gpu():
    try:
        import GPUtil
    except ImportError:
        wlog("please install GPUtil for automatically selecting GPU")
        return ""
    if len(GPUtil.getGPUs()) == 0:
        return ""
    id_list = GPUtil.getAvailable(order="load", maxLoad=0.7, maxMemory=0.7, limit=8)
    if len(id_list) == 0:
        print("GPU memory is not enough for predicted usage")
        raise NotImplementedError
    return str(id_list[0])


def set_file_logger(exp_logger, args):
    # use tensorboard + this function to substitute ExpSaver
    args_dict = vars(args)
    dir_path = args.dir_path
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, "para.json"), "w") as fp:
        json.dump(args_dict, fp, indent=4, sort_keys=True)
    logfile = os.path.join(dir_path, "exp.log")
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    exp_logger.addHandler(fh)
    if os.name != 'nt':
        signal.signal(signal.SIGQUIT, partial(rename_quit_handler, args))
        signal.signal(signal.SIGTERM, partial(delete_quit_handler, args))


def list_args(args):
    for e in sorted(vars(args).items()):
        print("args.%s = %s" % (e[0], e[1] if not isinstance(e[1], str) else '"%s"' % e[1]))


def form_dir_path(task, args):
    args_dict = vars(args)
    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    log_arg_list = []
    for e in args.log_arg.split("-"):
        v = args_dict.get(e, None)
        if v is None:
            log_arg_list.append(str(e))
        elif isinstance(v, str):
            log_arg_list.append(str(v))
        else:
            log_arg_list.append("%s=%s" % (e, str(v)))
    args.exp_marker = exp_marker = "-".join(log_arg_list)
    exp_marker = "%s/%s/%s@%s@%d" % (task, args.dataset, run_time, exp_marker, os.getpid())
    base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir
    dir_path = os.path.join(base_dir, exp_marker)
    return dir_path


def summary(data):
    assert isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
    wlog("shape: %s, num of points: %d, pixels: %d" % (str(data.shape), data.shape[0], np.prod(data.shape[1:])))
    wlog("max: %g, min %g" % (data.max(), data.min()))
    wlog("mean: %g" % data.mean())
    wlog("mean of abs: %g" % np.abs(data).mean())
    wlog("mean of square sum: %g" % (data ** 2).mean())


def remove_outliers(x, outlier_constant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    iqr = (upper_quartile - lower_quartile) * outlier_constant
    quartile_set = (lower_quartile - iqr, upper_quartile + iqr)

    result = a[np.where((a >= quartile_set[0]) & (a <= quartile_set[1]))]

    return result


def vis_step(writer, step, dicts):
    for k in dicts:
        writer.add_scalar(k, dicts[k], step)


def delete_quit_handler(g_var, signal, frame):
    shutil.rmtree(g_var.dir_path)
    sys.exit(0)


def rename_quit_handler(g_var, signal, frame):
    os.rename(g_var.dir_path, g_var.dir_path + "_stop")
    sys.exit(0)
