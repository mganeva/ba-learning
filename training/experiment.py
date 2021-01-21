import click
import json
from os import makedirs, path
from glob import glob

from tensorflow.keras import optimizers
from tensorflow.keras import models as k_models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from training import image_ops
from training.batch_generator import BatchGenerator
from training import lr
from training import models


def validate_helper(config, required, optional):
    missing_fields = set(required) - set(config.keys())

    if missing_fields:
        raise ValueError(
            "the following fields are missing: {}".format(missing_fields))

    extra_fields = set(config.keys()) - set(required) - set(optional)

    if extra_fields:
        raise ValueError(
            "the following fields are not recognized: {}".format(extra_fields))


def validate(filename):
    REQUIRED_FIELDS = [
        "model",
        "optimizer",
        "data",
        "loss",
        "train_epochs"
    ]

    OPTIONAL_FIELDS = [
        "lr_schedule",
        "use_snapshot",
        "early_stopping_patience"
    ]

    with open(filename, "r") as json_file:
        config = json.load(json_file)

    validate_helper(config, REQUIRED_FIELDS, OPTIONAL_FIELDS)
    validate_model(config["model"])
    validate_optimizer(config["optimizer"])
    validate_data(config["data"])

    if "lr_schedule" in config:
        validate_lr_schedule(config["lr_schedule"])

    if not config["loss"] in ["kullback_leibler_divergence"]:
        raise ValueError("loss '{}' is not recognized".format(config["loss"]))

    epochs = config["train_epochs"]

    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("'train_epochs' must be a positive integer")


def validate_model(model_config):
    validate_helper(model_config, ["name", "args"], [])
    validate_helper(model_config["args"], ["input_shape", "num_output"], [])


def validate_optimizer(optimizer_config):
    validate_helper(optimizer_config, ["name", "args"], [])


def validate_lr_schedule(lr_config):
    validate_helper(lr_config, ["name", "args"], [])


def validate_data(data_config):
    validate_helper(data_config,
                    ["train_dir", "val_dir", "batch_size"],
                    ["train_preprocessing", "val_preprocessing", "train_seed", "val_seed", "nproc", "max_train", "max_val"])


def get_default_callbacks(snapshot_dir, log_dir):
    if not path.exists(snapshot_dir):
        makedirs(snapshot_dir)

    if not path.exists(log_dir):
        makedirs(log_dir)

    snapshot_fmt = path.join(snapshot_dir, "model.{epoch:04d}.hdf5")
    callbacks = [
        ModelCheckpoint(snapshot_fmt, monitor='val_loss'),
        TensorBoard(log_dir=log_dir)
    ]
    return callbacks


def run_experiment(filename):
    validate(filename)

    with open(filename, "r") as json_file:
        config = json.load(json_file)

    model = config["model"]
    train_epochs = config["train_epochs"]
    early_stopping_patience = config.get("early_stopping_patience", None)
    data = config["data"]
    use_snapshot = config.get("use_snapshot", False)
    train_dir = data["train_dir"]
    val_dir = data["val_dir"]
    batch_size = data["batch_size"]
    nproc = data.get("nproc", None)
    max_train = data.get("max_train", None)
    max_val = data.get("max_val", None)

    train_seed = data.get("train_seed", None)
    val_seed = data.get("val_seed", None)

    train_preprocessing = data.get("train_preprocessing", [])
    val_preprocessing = data.get("val_preprocessing", [])

    train_ops, val_ops = [], []

    for stage in train_preprocessing:
        name = stage["name"]
        args = stage.get("args", {})
        train_ops.append(getattr(image_ops, name)(**args))

    for stage in val_preprocessing:
        name = stage["name"]
        args = stage.get("args", {})
        val_ops.append(getattr(image_ops, name)(**args))

    train_gen = BatchGenerator(
        train_dir, batch_size, train_ops, nproc, train_seed, max_train)
    val_gen = BatchGenerator(
        val_dir, batch_size, val_ops, nproc, val_seed, max_val)

    callbacks = get_default_callbacks("snapshots", "logs")
    with open("./logs/train_config.json", 'w') as json_file:
        json.dump(config, json_file, indent=4)

    lr_config = config.get("lr_schedule", None)

    initial_epoch = 0

    if lr_config:
        lr_args = lr_config.get("args", {})
        lr_args["epochs"] = train_epochs
        lr_args["steps_per_epoch"] = train_gen.steps_per_epoch
        lr_args["initial_epoch"] = initial_epoch

        print("Creating LR schedule '{}'".format(lr_config["name"]))
        lr_schedule = getattr(lr, lr_config["name"])(**lr_args)
        callbacks.append(lr_schedule)
    
    if early_stopping_patience is not None:
        callbacks.append(EarlyStopping(patience=early_stopping_patience))

    opt_config = config["optimizer"]
    opt = opt_config["name"]
    opt_args = opt_config.get("args", {})

    optimizer = getattr(optimizers, opt)(**opt_args)
    loss = config["loss"]

    model_name = model["name"]
    model_args = model.get("args", {})

    snapshots = sorted(glob("snapshots/model.*.hdf5"))

    if use_snapshot and len(snapshots) > 0:
        print("Loading model from snapshot '%s'" % snapshots[-1])
        model = k_models.load_model(snapshots[-1])
        initial_epoch = int(snapshots[-1].split(".")[-2])
        print("Last epoch was {:d}".format(initial_epoch))
    else:
        model = getattr(models, model_name)(**model_args)
        model.compile(optimizer=optimizer, loss=loss)
    print(model.summary())

    try:
        model.fit_generator(
            train_gen, epochs=train_epochs, steps_per_epoch=train_gen.steps_per_epoch,
            validation_data=val_gen, validation_steps=val_gen.steps_per_epoch,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            use_multiprocessing=False)
    except:
        train_gen.terminate()
        val_gen.terminate()
    finally:
        train_gen.close()
        val_gen.close()

    print("Training completed successfully")
