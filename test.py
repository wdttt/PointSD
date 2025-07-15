import os
import torch

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from pointbert.point_encoder import PointTransformer
from easydict import EasyDict
import yaml


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, split="train", process_data=False, npoint=1024):
        self.root = root
        self.npoints = npoint
        self.process_data = process_data
        self.uniform = False
        self.use_normals = False
        self.num_category = 40

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_test.txt"))
            ]
        else:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_test.txt"))
            ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (
                shape_names[i],
                os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt",
            )
            for i in range(len(shape_ids[split]))
        ]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints),
            )
        else:
            self.save_path = os.path.join(
                root,
                "modelnet%d_%s_%dpts.dat" % (self.num_category, split, self.npoints),
            )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(
                    "Processing data %s (only running in the first time)..."
                    % self.save_path
                )
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0 : self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0 : self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_ckpt(ckpt):
    new_ckpt = {}
    for k, v in ckpt.items():
        k = k.replace("module.", "")
        if "point_encoder." in k:
            k = k.replace("point_encoder.", "")
        new_ckpt[k] = v
    return new_ckpt


# Load the model
def validate_model(model, dtype, npoint=1024):
    device = "cuda"
    train = ModelNetDataLoader(
        "Point-MAE/data/ModelNet/modelnet40_normal_resampled/",
        split="train",
        npoint=npoint,
    )
    test = ModelNetDataLoader(
        "Point-MAE/data/ModelNet/modelnet40_normal_resampled/",
        split="test",
        npoint=npoint,
    )

    # Calculate the image features
    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for pc, labels in tqdm(DataLoader(dataset, batch_size=100, num_workers=8)):
                features, _ = model(pc.to(device=device, dtype=dtype))
                all_features.append(features)
                all_labels.append(labels)
        return (
            torch.cat(all_features).cpu().numpy(),
            torch.cat(all_labels).cpu().numpy(),
        )

    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
    return accuracy
    # print(f"Accuracy = {accuracy:.3f}")
    # ```

    # Note that the `C` value should be determined via a hyperparameter sweep using a validation split.
