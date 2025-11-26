#!/usr/bin/env python3
"""Train a ResNet50V2 model on the myna dataset and save a deployable model.

Creates `model.h5` (Keras HDF5) and a `labels.json` file listing class names.
The script downloads the dataset if not present, extracts it into `./data`,
and trains using `tf.keras.utils.image_dataset_from_directory`.

Usage examples:
  python training.py --epochs 10 --batch-size 8
  python training.py --data-dir ./data --output model.h5
"""

import os
import argparse
import json
import urllib.request
import zipfile
import shutil
import tempfile

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
import atexit

DEFAULT_URL = (
    "https://github.com/yenlung/Deep-Learning-Basics/raw/master/images/myna.zip"
)


def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    # download to a temporary directory and extract there, then move expected folders
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "myna.zip")
        print(f"Downloading dataset from {url} to {zip_path}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        print(f"Extracting {zip_path} to temporary directory {tmpdir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        # Move any class subfolders into dest_dir
        moved = False
        for entry in os.listdir(tmpdir):
            src = os.path.join(tmpdir, entry)
            dst = os.path.join(dest_dir, entry)
            if os.path.isdir(src):
                # If destination exists, skip moving
                if os.path.exists(dst):
                    print(f"Destination {dst} already exists; skipping move of {src}")
                else:
                    print(f"Moving {src} -> {dst}")
                    shutil.move(src, dst)
                    moved = True

        # If nothing was moved, perhaps the zip had files at top-level; try extracting there
        if not moved:
            # As a fallback, extract zip directly into dest_dir
            print(f"No folders detected in temporary extraction; extracting zip directly into {dest_dir}")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest_dir)


def build_model(num_classes=3):
    base = ResNet50V2(include_top=False, pooling="avg", input_shape=(256, 256, 3))
    base.trainable = False
    model = tf.keras.Sequential([base, tf.keras.layers.Dense(num_classes, activation="softmax")])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def get_dataset(data_dir, batch_size):
    # image_dataset_from_directory expects class subfolders inside data_dir
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
    )
    # capture class names before mapping/prefetch which create new dataset objects
    class_names = ds.class_names
    ds = ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y))
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, class_names


def main():
    parser = argparse.ArgumentParser(description="Train myna ResNet model and save for deployment")
    parser.add_argument("--data-dir", default="./data", help="Root directory for dataset (contains class subfolders)")
    parser.add_argument("--download-url", default=DEFAULT_URL, help="URL to download dataset zip if missing")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="model.h5", help="Output model file (HDF5)")
    parser.add_argument("--save-savedmodel", default="saved_model", help="Optional SavedModel folder to write")
    args = parser.parse_args()

    # Normalize data dir to absolute path to avoid TF path issues on Windows
    data_dir = os.path.abspath(args.data_dir)

    # Ensure dataset exists; if not, download & extract to data_dir
    expected_subfolders = ["crested_myna", "javan_myna", "common_myna"]
    need_download = not all(os.path.isdir(os.path.join(data_dir, s)) for s in expected_subfolders)
    if need_download:
        print("Dataset not found in the expected layout. Downloading and extracting...")
        os.makedirs(data_dir, exist_ok=True)
        download_and_extract(args.download_url, data_dir)
    else:
        print(f"Dataset folders appear present in {data_dir}; will use existing files.")

    # Verify data_dir exists before passing to TF
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # TensorFlow's file API on Windows sometimes cannot read paths with
    # non-ASCII characters (OneDrive paths). If TF can't see the directory
    # but the OS can, copy the dataset into a temporary directory with an
    # ASCII-only path and use that for loading.
    dataset_path = data_dir
    try:
        if not tf.io.gfile.exists(dataset_path):
            # create a temporary copy
            tmpdir = tempfile.mkdtemp(prefix="myna_data_")
            print(f"TensorFlow cannot access {dataset_path}; copying data to temporary dir {tmpdir}")
            for name in os.listdir(data_dir):
                src = os.path.join(data_dir, name)
                dst = os.path.join(tmpdir, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
            dataset_path = tmpdir
            # schedule cleanup
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: TF file API check failed: {e}; attempting to use original path anyway")

    # Create dataset (robustly handle TF inability to access OneDrive paths)
    try:
        ds, class_names = get_dataset(dataset_path, args.batch_size)
    except Exception as e:
        # If TF cannot read the directory (e.g. NotFoundError on Windows/OneDrive),
        # copy class subfolders to an ASCII-only temporary directory and retry.
        msg = str(e)
        if isinstance(e, tf.errors.NotFoundError) or "Could not find directory" in msg or "NotFoundError" in msg:
            tmpdir = tempfile.mkdtemp(prefix="myna_data_")
            print(f"TensorFlow raised error reading {dataset_path}: {e}\nCopying data to temporary dir {tmpdir} and retrying...")
            for name in os.listdir(data_dir):
                src = os.path.join(data_dir, name)
                dst = os.path.join(tmpdir, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
            # schedule cleanup
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
            dataset_path = tmpdir
            ds, class_names = get_dataset(dataset_path, args.batch_size)
        else:
            raise

    print("Detected classes:", class_names)

    model = build_model(num_classes=len(class_names))
    model.summary()

    print("Training model...")
    model.fit(ds, epochs=args.epochs)

    # Save model in HDF5 for portability
    print(f"Saving model to {args.output}...")
    model.save(args.output)

    # Also save SavedModel folder (optional)
    print(f"Saving SavedModel to {args.save_savedmodel}...")
    model.save(args.save_savedmodel)

    # Save class names for inference apps
    labels_path = os.path.splitext(args.output)[0] + "_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Saved labels to {labels_path}")

    print("Done. Commit `model.h5`, labels JSON and optionally saved_model folder to GitHub for deployment.")


if __name__ == "__main__":
    main()
