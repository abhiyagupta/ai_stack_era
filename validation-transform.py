import argparse
import os
import shutil







import os
import shutil

# Hardcoded paths
DATA_DIR = "./Imagenet/ILSVRC/Data/CLS-LOC/val"
LABELS_FILE = "./Imagenet/ILSVRC/LOC_val_solution.csv"

processed_classes = set()

# Open the validation labels file
with open(LABELS_FILE, "r") as file:
    # Skip the header
    next(file)
    for line in file:
        img_name, labels = line.strip().split(",")
        class_name = labels.split(" ")[0]

        # Create a directory for this class if it doesn't exist
        dir_path = os.path.join(DATA_DIR, class_name)
        if class_name not in processed_classes:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            processed_classes.add(class_name)

        # Move the image to the corresponding class folder
        src_path = os.path.join(DATA_DIR, f"{img_name}.JPEG")
        dest_path = os.path.join(dir_path, f"{img_name}.JPEG")
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"Image not found: {src_path}")


# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--dir", help="dir with the images", required=True)
# parser.add_argument("-l", "--labels", help="file with image name to class label mapping", required=True)

# args = parser.parse_args()

# processed_classes = set()

# with open(args.labels, "r") as file:
#     # skip header
#     next(file)
#     for line in file:
#         img_name, labels = line.split(",")
#         class_name = labels.split(" ")[0]
#         # create a dir for this classname
#         if class_name not in processed_classes:
#             dir_path = args.dir + "/" + class_name
#             if not os.path.exists(dir_path):
#                 os.mkdir(dir_path)
#         shutil.move(args.dir + "/" + img_name + ".JPEG", args.dir + "/" + class_name+ "/" + img_name + ".JPEG")
