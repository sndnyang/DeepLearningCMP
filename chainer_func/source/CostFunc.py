import numpy as np
from chainer import Variable, cuda
import chainer.functions as F
from .chainer_functions import loss


XI = 1e-6


def loss_labeled(forward, x, t, args):
    y = forward(x, update_batch_stats=True)
    L = F.softmax_cross_entropy(y, Variable(t))
    if 'lat' in args.method:
        one_hot = np.zeros(y.shape).astype(np.float32)
        one_hot[np.arange(y.shape[0]), cuda.to_cpu(t)] = 1
        L += loss.at_loss(forward, x, Variable(cuda.to_gpu(one_hot, device=t.device)), train=True, epsilon=args.at_epsilon)
    return L


def loss_unlabeled(forward, x, y, args, lx=None, ly=None):
    if 'vat' in args.method:
        # Virtual adversarial training loss
        logit = forward(x, train=True, update_batch_stats=False)
        if args.method == "vat_p":
            vat_loss = loss.vat_loss(forward, loss.distance, x, y, epsilon=args.epsilon, xi=XI, p_logit=logit.data)
            l_logit = forward(lx, train=True, update_batch_stats=False)
            vat_loss += args.reg_lamb * loss.vat_plus(forward, loss.distance, lx, ly, epsilon=args.epsilon, xi=XI, p_logit=l_logit.data)
        elif args.method == "vat_pp":
            vat_loss = loss.vat_double(forward, loss.distance, x, y, epsilon=args.epsilon, neg_eps=args.neg_eps,
                                       reg_lamb=args.reg_lamb, xi=XI, p_logit=logit.data)
            l_logit = forward(lx, train=True, update_batch_stats=False)
            vat_loss += loss.vat_pp(forward, loss.distance, lx, ly, epsilon=args.epsilon, neg_eps=args.neg_eps,
                                    correct_lamb=args.correct_lamb, xi=XI, p_logit=l_logit.data)
        elif args.method == "vat_p_m":
            vat_loss = loss.vat_plus(forward, loss.distance, x, y, epsilon=args.epsilon, xi=XI, p_logit=logit.data, dif=True)
        elif args.method == "vat_pp_m":
            vat_loss = loss.vat_pp(forward, loss.distance, x, y, epsilon=args.epsilon, neg_eps=args.neg_eps,
                                   correct_lamb=args.reg_lamb, xi=XI, p_logit=logit.data, dif=True)
        elif args.method == "vat_rev":
            vat_loss = loss.vat_rev(forward, loss.distance, x, epsilon=args.epsilon, reg_lamb=args.reg_lamb, xi=XI, power_iter=args.power_iter,
                                    p_logit=logit.data)
        elif args.method == "vat_rev_d":
            vat_loss = loss.vat_rev_double(forward, loss.distance, x, epsilon=args.epsilon, neg_eps=args.neg_eps, reg_lamb=args.reg_lamb,
                                           trade_lamb=args.trade_lamb, xi=XI, power_iter=args.power_iter, p_logit=logit.data)
        elif "vat_d" in args.method:
            vat_loss = loss.vat_double(forward, loss.distance, x, epsilon=args.epsilon, neg_eps=args.neg_eps, reg_lamb=args.reg_lamb,
                                       trade_lamb=args.trade_lamb, xi=XI, p_logit=logit.data)
        elif args.method == "vat_s":
            vat_loss = loss.vat_sharp(forward, loss.distance, x, epsilon=args.epsilon, neg_eps=args.neg_eps, reg_lamb=args.reg_lamb,
                                      trade_lamb=args.trade_lamb, xi=XI, p_logit=logit.data)
        elif args.method == "vat_n":
            vat_loss = loss.vat_noise(forward, loss.distance, x, epsilon=args.epsilon, noise_eps=args.neg_eps,
                                      reg_lamb=args.reg_lamb, xi=XI, p_logit=logit.data)
        else:
            vat_loss = args.reg_lamb * loss.vat_loss(forward, loss.distance, x, epsilon=args.epsilon, xi=XI, p_logit=logit.data)
        if 'ent' in args.method:
            ent_y_x = loss.entropy_y_x(logit)
            vat_loss += ent_y_x
        # Virtual adversarial training loss + Conditional Entropy loss
        return vat_loss
    elif args.method == 'baseline' or args.method == "lat":
        xp = cuda.get_array_module(x.data)
        return Variable(xp.array(0, dtype=xp.float32))
    else:
        raise NotImplementedError


def loss_test(forward, x, t):
    logit = forward(x, train=False)
    L, acc = F.softmax_cross_entropy(logit, t).data, F.accuracy(logit, t).data
    return L, acc
