#!/usr/bin/env python3
"""Simple training script extracted from `colab.ipynb`.

Usage:
  python scripts/train.py --data-dir /path/to/data --save-path models/myna_model --epochs 10

Expects `data_dir` to contain one subdirectory per class, e.g.
  data/crested_myna/*.jpg
  data/javan_myna/*.jpg
  data/common_myna/*.jpg

The script saves a Keras SavedModel at `--save-path` and a `labels.json` file next to it.
"""
import os
import argparse
import json
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def load_dataset(data_dir, target_size=(256, 256)):
    data = []
    labels = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_dir}")
    for i, cls in enumerate(classes):
        thedir = os.path.join(data_dir, cls)
        for fname in os.listdir(thedir):
            path = os.path.join(thedir, fname)
            try:
                img = load_img(path, target_size=target_size)
                x = img_to_array(img)
                data.append(x)
                labels.append(i)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, classes


def build_model(num_classes):
    base = ResNet50V2(include_top=False, pooling="avg", weights="imagenet")
    base.trainable = False
    model = Sequential()
    model.add(base)
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Path to dataset root (subfolders per class)")
    parser.add_argument("--save-path", default="models/myna_model", help="Path to save the Keras SavedModel")
    parser.add_argument("--labels-path", default="models/labels.json", help="Path to save class labels (JSON)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=23)
    args = parser.parse_args()

    print("Loading dataset from", args.data_dir)
    x, y, classes = load_dataset(args.data_dir)
    print(f"Loaded {len(x)} images across {len(classes)} classes: {classes}")

    x = preprocess_input(x)
    y_cat = to_categorical(y, num_classes=len(classes))

    model = build_model(len(classes))
    model.summary()

    model.fit(x, y_cat, batch_size=args.batch_size, epochs=args.epochs)

    save_dir = args.save_path
    # ensure parent directory exists (handle both file and dir paths)
    parent_dir = os.path.dirname(save_dir) or save_dir
    os.makedirs(parent_dir, exist_ok=True)
    print("Saving model to", save_dir)
    # If save path ends with a Keras file extension, save as single-file Keras format
    import tensorflow as _tf
    from tensorflow.keras.models import save_model as keras_save_model

    try:
        if save_dir.lower().endswith(('.keras', '.h5')):
            model.save(save_dir)
        else:
            # Prefer Keras' save_model with TF format (handles Keras objects correctly)
            try:
                keras_save_model(model, save_dir, save_format='tf')
            except TypeError:
                # Older/newer TF/Keras combos may not accept save_format kw; try model.save with dir
                try:
                    model.save(save_dir)
                except Exception:
                    # Last resort: use low-level SavedModel export
                    _tf.saved_model.save(model, save_dir)
    except Exception as e:
        print("Error while saving model:", repr(e))
        # Fallback: save as single-file Keras format
        alt = save_dir + '.keras'
        try:
            print("Falling back to saving as single-file Keras format:", alt)
            model.save(alt)
            print("Saved fallback model to", alt)
        except Exception as e2:
            print("Final save attempt failed:", repr(e2))
            raise

    os.makedirs(os.path.dirname(args.labels_path), exist_ok=True)
    with open(args.labels_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)
    print("Saved labels to", args.labels_path)


if __name__ == "__main__":
    main()
