import os
import glob
import argparse
from breeds_helpers import setup_breeds
from breeds_helpers import *
from breeds_helpers import ClassHierarchy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="living17", help="BREEDS Dataset to create symlink dataset")
    args = parser.parse_args()

    imnet_dir = '/data/datasets/image_datasets/imagenet1k-ILSVRC2012'
    imnet_train_dir = os.path.join(imnet_dir, "train")
    imnet_val_dir = os.path.join(imnet_dir, "val")
    info_dir = './imagenet_class_hierarchy/modified'
    sym_dir = "/data/datasets/image_datasets/BREEDS-Benchmarks/symlink_datasets"

    hier = ClassHierarchy(info_dir)

    if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
        print("Downloading class hierarchy information into `info_dir`")
        setup_breeds(info_dir)

    if args.dataset == "living17":
        superclasses, subclass_split, label_map, subclass_ids = make_living17(info_dir, split="rand") 
    elif args.dataset == "entity13":
        superclasses, subclass_split, label_map, subclass_ids = make_entity13(info_dir, split="rand")
    elif args.dataset == "entity30":
        superclasses, subclass_split, label_map, subclass_ids = make_entity30(info_dir, split="rand")
    elif args.dataset == "nonliving26":
        superclasses, subclass_split, label_map, subclass_ids = make_nonliving26(info_dir, split="rand")
    else:
        raise ValueError("Invalid dataset name")

    source_ids, target_ids = subclass_ids

    # first create directories for every class in label_map values
    ds_dir = os.path.join(sym_dir, args.dataset)
    src_dir = os.path.join(ds_dir, "source")
    tgt_dir = os.path.join(ds_dir, "target")

    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    
    src_train_dir = os.path.join(src_dir, "train")
    src_val_dir = os.path.join(src_dir, "val")

    os.makedirs(src_train_dir, exist_ok=True)
    os.makedirs(src_val_dir, exist_ok=True)

    for label in label_map.values():
        os.makedirs(os.path.join(src_train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(src_val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(tgt_dir, label), exist_ok=True)

    n_classes = len(label_map)

    for i_class in range(n_classes):

        new_class = label_map[i_class]
        print("Creating symlinks for class: ", new_class)

        # for every image file in folders of orig_classes, create a symlink in src_dir in new_class folder
        orig_classes = source_ids[i_class]

        # grab images from the training set
        for orig_class in orig_classes:
            orig_class_dir = os.path.join(imnet_train_dir, orig_class)
            orig_files = glob.glob(os.path.join(orig_class_dir, "*.JPEG"))
            for orig_file in orig_files:
                file_name = os.path.basename(orig_file)
                new_file_name = f"{new_class}_{hash(file_name)}.JPEG"
                new_file_path = os.path.join(src_train_dir, new_class, new_file_name)
                os.symlink(orig_file, new_file_path)

        # do the same for images from the validation set
        for orig_class in orig_classes:
            orig_class_dir = os.path.join(imnet_val_dir, orig_class)
            orig_files = glob.glob(os.path.join(orig_class_dir, "*.JPEG"))
            for orig_file in orig_files:
                file_name = os.path.basename(orig_file)
                new_file_name = f"{new_class}_{hash(file_name)}.JPEG"
                new_file_path = os.path.join(src_val_dir, new_class, new_file_name)
                os.symlink(orig_file, new_file_path)

        # now build the target dataset using the target_ids and images from validation set
        orig_classes = target_ids[i_class]
        for orig_class in orig_classes:
            orig_class_dir = os.path.join(imnet_val_dir, orig_class)
            orig_files = glob.glob(os.path.join(orig_class_dir, "*.JPEG"))
            for orig_file in orig_files:
                file_name = os.path.basename(orig_file)
                new_file_name = f"{new_class}_{hash(file_name)}.JPEG"
                new_file_path = os.path.join(tgt_dir, new_class, new_file_name)
                os.symlink(orig_file, new_file_path)



    
        

