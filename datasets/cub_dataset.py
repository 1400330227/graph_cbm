import json
import os
from os.path import join, isdir, isfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.utils import classes

from datasets import transforms


class CubDataset(Dataset):
    def __init__(self, root, transforms, is_train):
        self.root = root
        self.transforms = transforms
        self.img_root = os.path.join(self.root, "images")
        self.relationship_root = os.path.join(self.root, "relations")
        self.annotations_root = os.path.join(self.root, "images.txt")
        self.classes_root = os.path.join(self.root, "classes.txt")
        self.train_test_split_root = os.path.join(self.root, "train_test_split.txt")
        self.json_file = os.path.join(self.root, "cub_attributes.json")
        self.predicate_file = os.path.join(self.root, "predicate.json")
        self.is_train = is_train
        self.train_val_data, test_data = [], []
        self.class_dict = {}
        self.predicate_dict = {}
        path_to_id_map = dict()
        with open(self.annotations_root) as f:
            for line in f:
                items = line.strip().split()
                path_to_id_map[join(self.img_root, items[1])] = int(items[0])

        classes_label_to_id_map = dict()
        with open(self.classes_root) as f:
            for line in f:
                items = line.strip().split()
                classes_label_to_id_map[items[1]] = int(items[0])

        is_train_test = dict()
        with open(self.train_test_split_root) as f:
            for line in f:
                idx, is_train = line.strip().split()
                is_train_test[int(idx)] = int(is_train)

        with open(self.json_file, 'r') as f:
            self.class_dict = json.load(f)

        with open(self.predicate_file, 'r') as f:
            self.predicate_dict = json.load(f)

        folder_list = [f for f in os.listdir(self.img_root) if isdir(join(self.img_root, f))]
        folder_list.sort()
        self.data = []
        for i, folder in enumerate(folder_list):
            folder_path = join(self.img_root, folder)
            classfile_list = [cf for cf in os.listdir(folder_path)
                              if (isfile(join(folder_path, cf)) and cf[0] != '.')
                              and cf.lower().endswith('.jpg')]
            for cf in classfile_list:
                img_id = path_to_id_map[(folder_path + '/' + cf)]
                img_path = (folder + '/' + cf)
                classes_label = classes_label_to_id_map[folder]
                if self.is_train and is_train_test[img_id]:
                    self.data.append((img_id, img_path, classes_label))
                else:
                    self.data.append((img_id, img_path, classes_label))

    def __len__(self):
        # self.data = self.data[:100]
        return len(self.data)

    def __getitem__(self, idx):
        img_id, img_path, class_label = self.data[idx]
        relationship_path = os.path.join(self.relationship_root, img_path)
        img_path = os.path.join(self.img_root, img_path)
        json_path = img_path.replace(".jpg", ".json")
        relationship_path = relationship_path.replace(".jpg", ".json")
        image = Image.open(img_path)
        with open(json_path) as f:
            json_data = json.load(f)
        boxes = []
        labels = []
        iscrowd = []
        for shape in json_data["shapes"]:
            label = shape["label"]
            points = np.array(shape["points"], dtype=np.float32)
            x_min, y_min = points[0]
            x_max, y_max = points[2]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_dict[label])
            iscrowd.append(int(shape["difficult"]))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        class_label = torch.as_tensor([class_label], dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        relations, relation_tuple = self.get_relation_map(relationship_path, boxes)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["relation"] = relations
        # if relationship_path == 'data/CUB_200_2011/relations/011.Rusty_Blackbird/Rusty_Blackbird_0032_6611.json':
        # self.set_relation_map(relationship_path, boxes, labels)
        target["relation_tuple"] = relation_tuple
        target["class_label"] = class_label
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def get_relation_map(self, relationship_path, obj_boxes):
        num_objs = obj_boxes.shape[0]
        with open(relationship_path) as f:
            relationships_data = json.load(f)
        _relations = np.array(relationships_data["relationships"])
        _relation_predicates = np.array(relationships_data["predicates"])
        relations = torch.zeros((num_objs, num_objs), dtype=torch.int64)
        valid_relations = []
        for idx in range(len(_relations)):
            s_idx, o_idx = _relations[idx]
            predicate = _relation_predicates[idx]
            if s_idx < num_objs and o_idx < num_objs:
                relations[s_idx, o_idx] = predicate
                valid_relations.append([s_idx, o_idx, predicate])
        if valid_relations:
            relation_tuple = torch.tensor(valid_relations, dtype=torch.int64)
        else:
            relation_tuple = torch.zeros((0, 3), dtype=torch.int64)
        return relations, relation_tuple

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def coco_index(self, idx):
        img_id, img_path, class_label = self.data[idx]
        img_path = os.path.join(self.img_root, img_path)
        json_path = img_path.replace(".jpg", ".json")
        with open(json_path) as f:
            json_data = json.load(f)

        data_height = int(json_data["imageHeight"])
        data_width = int(json_data["imageWidth"])
        boxes = []
        labels = []
        iscrowd = []
        for shape in json_data["shapes"]:
            label = shape["label"]
            points = np.array(shape["points"], dtype=np.float32)
            x_min, y_min = points[0]
            x_max, y_max = points[2]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_dict[label])
            iscrowd.append(int(shape["difficult"]))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        class_label = torch.as_tensor(class_label, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["relation"] = torch.randint(0, 51, [len(labels), len(labels)])
        target["class_label"] = class_label

        return (data_height, data_width), target

    def set_relation_map(self, relationship_path, obj_boxes, labels):
        num_objs = obj_boxes.shape[0]
        with open(relationship_path) as f:
            try:
                relationships_data = json.load(f)
            except json.decoder.JSONDecodeError:
                print(relationship_path)
        label_to_indices = {}
        obj_relations = torch.zeros((num_objs, num_objs), dtype=torch.int64)
        obj_relation_triplets = []
        for idx, label in enumerate(labels):
            label = label.item()
            label_to_indices.setdefault(label, []).append(idx)

        relationships_data["relationships"] = []
        relationships_data["predicates"] = []
        for rel_pair in relationships_data["rel_pairs"]:
            obj_start, rel, obj_end = rel_pair[0], rel_pair[1], rel_pair[2]
            obj_start_label = self.class_dict.get(obj_start, -1)
            obj_end_label = self.class_dict.get(obj_end, -1)
            rel_label = self.predicate_dict.get(rel, 0)
            if obj_start_label == -1 or obj_end_label == -1:
                continue
            obj_relation_triplets.append((obj_start_label, obj_end_label, rel_label))
            start_indices = label_to_indices.get(obj_start_label, [])
            end_indices = label_to_indices.get(obj_end_label, [])
            if start_indices and end_indices:
                rows = torch.tensor(start_indices, dtype=torch.long)
                cols = torch.tensor(end_indices, dtype=torch.long)
                grid_i = rows.view(-1, 1)
                grid_j = cols.view(1, -1)

                current_block = obj_relations[grid_i, grid_j]
                mask = current_block == 0
                updated_block = torch.where(mask, rel_label, current_block)
                obj_relations[grid_i, grid_j] = updated_block

                for i in range(len(rows)):
                    for j in range(len(cols)):
                        relationships_data["relationships"].append([
                            rows[i].item(),
                            cols[j].item()
                        ])
                        relationships_data["predicates"].append(
                            updated_block[i, j].item()
                        )
        try:
            with open(relationship_path, 'w') as f:
                json.dump(relationships_data, f, indent=2)
        except IOError as e:
            print(f"Error writing to {relationship_path}: {str(e)}")


if __name__ == '__main__':
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    cub_dataset = CubDataset("data/CUB_200_2011", data_transform["train"], True)

    train_data_loader = torch.utils.data.DataLoader(cub_dataset, batch_size=2, shuffle=True,
                                                    collate_fn=cub_dataset.collate_fn)

    for i, (img, target) in enumerate(train_data_loader):
        print(len(img))
