import os
import re
import sys
import json
import time
import shutil
import random
import logging
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
wlog = logger.info
exp_seed = random.randrange(sys.maxsize) % 10000


def set_framework_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cmp_para(params, keys):
    suffix = ""
    for e in keys:
        assert e in params
        suffix += "_%s=%s" % (e, str(params[e] if not isinstance(params, float) else "%g" % params[e]))
    return suffix


class ExpSaver:

    def __init__(self, method, args, dir_keys=None, file_keys=None):
        if dir_keys is None:
            dir_keys = []
        if file_keys is None:
            file_keys = []
        assert method is not None
        self.run_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        args.method = method
        # sometimes, the code is using docopt library.
        if not isinstance(args, dict):
            self.args = args
        else:
            self.args = argparse.Namespace()
            self.args.__dict__.update(args)
        self.trainer, self.task = method.split("-")[:2]
        self.exp_marker = get_cmp_para(args.__dict__, file_keys)
        self.log_dir = self.set_log_path(method, dir_keys)
        self.exp_seed = exp_seed
        self.save_params()
        self.keys = file_keys
        self.writer = None

    def update_args(self, args):
        self.args = args
        self.exp_marker = get_cmp_para(args.__dict__, self.keys)

    def set_log_path(self, method=None, tuned=None):
        base = os.path.join(os.environ["HOME"], "project", "results")

        args = self.args.__dict__
        marker = method + "-" + (args.get("dataset") if "dataset" in args else "toy")
        if tuned:
            marker += "-".join("%s=%s" % (e, str(self.args.__dict__.get(e))) for e in tuned)
            log_dir = os.path.join(base, "%s-%s_running" % (marker, self.run_time))
        else:
            log_dir = os.path.join(base, "%s-%s_running" % (marker, self.run_time))
        log_name = log_dir + '/logs_%s_%s.log' % (self.run_time, exp_seed)

        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, "npy"))
        os.makedirs(os.path.join(log_dir, "txt"))
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return log_dir

    def delete_dir(self):
        shutil.rmtree(self.log_dir)

    def finish_exp(self, keep_args=None):
        if keep_args is None:
            keep_args = {}
        if "epochs" in self.args and (self.args.epochs < 50 or (self.args.dataset in ["cifar10", "cifar100"] and self.args.epochs < 20)):
            shutil.rmtree(self.log_dir)
            return
        if "iterations" in self.args and self.args.iterations < 100:
            shutil.rmtree(self.log_dir)
            return
        args_dict = vars(self.args)
        for k in keep_args:
            if k in self.args and args_dict[k] < keep_args[k]:
                self.delete_dir()
                return
        wlog("args_list in this experiment %s", '\n'.join(str(e) for e in sorted(args_dict.items())))
        os.rename(self.log_dir, self.log_dir[:-8])

    def save_params(self):
        params = {}
        for e in self.args.__dict__:
            val = self.args.__dict__[e]
            if not isinstance(val, (int, float, bool, tuple, str, list, dict)):
                continue
            params[e] = val

        if self.log_dir:
            with open(os.path.join(self.log_dir, "params.json"), "w") as fp:
                json.dump(params, fp, indent=4)

        return params

    def save_hist(self, hist, metric="acc"):
        for e in ["train", "val", "test"]:
            if e not in hist:
                continue
            loss_array = np.array([e[1] for e in hist.get(e)])
            self.save_npy(loss_array, name="%s_loss_" % e)
            array = np.array([e[0] for e in hist.get(e)])
            self.save_npy(array, name="%s_%s_" % (e, metric))

    def save_npy(self, array, name=""):
        # make sure the log file and experimental results will save to different directories
        marker = (self.run_time, exp_seed)

        # default way is to save as .npy file, however I like save as a text file.
        # so save both.
        if self.args.exp == "avg":
            file_name = os.path.join(self.log_dir, "%s%s.npy" % (name, self.exp_marker))
        else:
            file_name = os.path.join(self.log_dir, "npy", "%s%s.npy" % (name, self.exp_marker))
        np.save(file_name, array)
        file_name = os.path.join(self.log_dir, "txt", "%s%s.npy.txt" % (name, self.exp_marker))
        np.savetxt(file_name, array[np.newaxis], fmt='%g', delimiter=',')
        return marker

    def save_figure(self, figure, name=""):
        figure.savefig(os.path.join(self.log_dir, "exp_results_%s%s.png" % (self.exp_marker, name)))

    def init_writer(self):
        p = exp_seed
        dir_marker = "%s_%s_%s_%s_%d" % (self.task, self.args.dataset, str(self.args.exp), self.run_time, p)
        dir_path = os.path.join(os.environ['HOME'], 'project/runs', dir_marker)
        self.writer = SummaryWriter(log_dir=dir_path)


class ExpLoader:
    """
    TODO:
        useless
    """
    def __init__(self, path):
        self.path = path
        self.args = {}
        self.load(path)

    def load(self, path):
        args = {}
        data_set, size, method = path.split("/")[5:8]
        args["data-set"] = data_set
        args["data-size"] = size
        args["method"] = method
        args["metrics"] = set([])

        for f in os.listdir(path):
            if not f.endswith(".npy"):
                continue
            parts = f[:-4].split("_")
            for e in parts:
                if "=" in e:
                    continue
                args["metrics"].add(e)
            for e in parts:
                if "=" not in e:
                    continue
                key, value = e.split("=")
                if key not in args:
                    args[key] = set([])
                args[key].add(value)
        self.args = args
        return args

    def avg_experiments(self):
        for f in os.listdir(self.path + "/npy"):
            if not f.endswith(".npy"):
                continue
            if "exp=1_" not in f and "exp=0_" not in f:
                continue
            f = re.sub("exp=[01]_", "exp=avg_", f)
            if os.path.isfile(os.path.join(self.path, f)):
                print("%s exists" % f)
                continue
            acc_list = []
            for e in self.args.get("exp", []):
                if e == "avg":
                    continue
                exp_f = f.replace("exp=avg_", "exp=%s_" % e)
                exp_f = os.path.join(self.path, "npy", exp_f)
                if not os.path.isfile(exp_f):
                    continue
                acc_list.append(np.load(exp_f))
            if len(acc_list) == 0:
                continue
            array = np.array(acc_list)
            array = np.mean(array, axis=0)

            np.save(os.path.join(self.path, f), array)
            np.savetxt(os.path.join(self.path, "txt", f), array[np.newaxis], fmt='%g', delimiter=',')
