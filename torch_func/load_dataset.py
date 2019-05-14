import os
import numpy as np


def load_npz_as_dict(path):
    data = np.load(path)
    return {key: data[key] for key in data}


def augmentation(images, random_crop=True, random_flip=True):
    # random crop and random flip
    h, w = images.shape[2], images.shape[3]
    pad_size = 2
    aug_images = []
    padded_images = np.pad(images, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'reflect')
    for image in padded_images:
        if random_flip:
            image = image[:, :, ::-1] if np.random.uniform() > 0.5 else image
        if random_crop:
            offset_h = np.random.randint(0, 2 * pad_size)
            offset_w = np.random.randint(0, 2 * pad_size)
            image = image[:, offset_h:offset_h + h, offset_w:offset_w + w]
        else:
            image = image[:, pad_size:pad_size + h, pad_size:pad_size + w]
        aug_images.append(image)
    ret = np.stack(aug_images)
    assert ret.shape == images.shape
    return ret


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.index = np.arange(self.N)

    @property
    def N(self):
        return len(self.data)

    def get(self, n=None, shuffle=True, aug_trans=False, aug_flip=False):
        if shuffle:
            ind = np.random.permutation(self.data.shape[0])
        else:
            ind = np.arange(self.data.shape[0])
        if n is None:
            n = self.data.shape[0]
        index = ind[:n]
        batch_data = self.data[index]
        batch_label = self.label[index]
        if aug_trans or aug_flip:
            assert batch_data.ndim == 4
            # shape of `image' [N, K, W, H]
            batch_data = augmentation(batch_data, aug_trans, aug_flip)
        return batch_data, batch_label


def load_dataset(dir_path, valid=False, dataset_seed=1):
    if dataset_seed > 5:
        # 1-5 -> [1, 5],  >5 -> [1, 5]
        dataset_seed = dataset_seed % 5 + 1
    if valid:
        train_l = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'labeled_train_valid.npz'))
        train_ul = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'unlabeled_train_valid.npz'))
        test = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'test_valid.npz'))
    else:
        train_l = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'labeled_train.npz'))
        train_ul = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'unlabeled_train.npz'))
        test = load_npz_as_dict(os.path.join(dir_path, 'seed' + str(dataset_seed), 'test.npz'))
    train_l['images'] = train_l['images'].reshape(train_l['images'].shape[0], 3, 32, 32).astype(np.float32)
    train_ul['images'] = train_ul['images'].reshape(train_ul['images'].shape[0], 3, 32, 32).astype(np.float32)
    test['images'] = test['images'].reshape(test['images'].shape[0], 3, 32, 32).astype(np.float32)
    train_l['labels'] = train_l['labels'].astype(np.int32)
    train_ul['labels'] = train_ul['labels'].astype(np.int32)
    test['labels'] = test['labels'].astype(np.int32)
    return Data(train_l['images'], train_l['labels']), Data(train_ul['images'], train_ul['labels']), Data(test['images'], test['labels'])
