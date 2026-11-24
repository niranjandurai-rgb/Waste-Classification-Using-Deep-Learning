"""Waste classification (cardboard/glass/metal/paper/plastic/trash).
Usage: python train.py --data dataset-resized --model mobilenet
       python train.py --data dataset-resized --model cnn
"""
import argparse, hashlib, json, pathlib
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG, SEED = 224, 42

def collect(root):
    paths, labels, seen = [], [], set()
    for i, c in enumerate(CLASSES):
        for p in sorted((root / c).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            h = hashlib.md5(p.read_bytes()).hexdigest()   # drop exact duplicates
            if h in seen:
                continue
            seen.add(h); paths.append(str(p)); labels.append(i)
    return np.array(paths), np.array(labels)

def make_ds(paths, labels, batch, shuffle):
    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        return img, y
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)

def augment():  # active only during training
    return keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
                             layers.RandomZoom(0.1), layers.RandomBrightness(0.1, value_range=(0, 255))])

def build_cnn():
    i = keras.Input((IMG, IMG, 3))
    x = augment()(i); x = layers.Rescaling(1 / 255)(x)
    for f in (32, 64, 128, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.4)(x)
    return keras.Model(i, layers.Dense(len(CLASSES), activation="softmax")(x)), None

def build_mobilenet():
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    base.trainable = False
    i = keras.Input((IMG, IMG, 3))
    x = augment()(i); x = layers.Rescaling(1 / 127.5, offset=-1)(x)   # MobileNetV2 normalisation
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.3)(x)
    return keras.Model(i, layers.Dense(len(CLASSES), activation="softmax")(x)), base

def compile_(m, lr):
    m.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True); a.add_argument("--model", choices=["cnn", "mobilenet"], default="mobilenet")
    a.add_argument("--epochs", type=int, default=15); a.add_argument("--batch", type=int, default=32)
    a.add_argument("--out", default="outputs")
    a = a.parse_args()
    out = pathlib.Path(a.out) / a.model; out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    paths, labels = collect(pathlib.Path(a.data))
    print("Images after de-duplication:", len(paths), dict(zip(CLASSES, np.bincount(labels))))
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    train, val, test = (make_ds(p_tr, y_tr, a.batch, True), make_ds(p_va, y_va, a.batch, False),
                        make_ds(p_te, y_te, a.batch, False))
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(6), y=y_tr)))

    model, base = build_cnn() if a.model == "cnn" else build_mobilenet()
    ckpt = str(out / "best_model.keras")
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True),
           keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]
    compile_(model, 1e-3 if a.model == "cnn" else 1e-3)
    hist = model.fit(train, validation_data=val, epochs=a.epochs, class_weight=cw, callbacks=cbs)
    acc, vacc = hist.history["accuracy"], hist.history["val_accuracy"]
    if base is not None:   # fine-tune top layers with small LR
        base.trainable = True
        for l in base.layers[:-30]: l.trainable = False
        compile_(model, 1e-5)
        h2 = model.fit(train, validation_data=val, epochs=8, class_weight=cw, callbacks=cbs)
        acc += h2.history["accuracy"]; vacc += h2.history["val_accuracy"]

    best = keras.models.load_model(ckpt)
    y_pred = np.argmax(best.predict(test), axis=1)
    report = classification_report(y_te, y_pred, target_names=CLASSES, output_dict=True)
    print(classification_report(y_te, y_pred, target_names=CLASSES))
    json.dump(report, open(out / "metrics.json", "w"), indent=2)
    json.dump(CLASSES, open(out / "classes.json", "w"))

    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=CLASSES).plot(xticks_rotation=45)
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()
    plt.plot(acc, label="train"); plt.plot(vacc, label="val"); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend(); plt.savefig(out / "curves.png", dpi=150)
    print("Saved to", out)

if __name__ == "__main__":
    main()
