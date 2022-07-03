from vgg16.architecture import VGG16
from tools.data import *
from tools.log import *
from vgg16 import *


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--name", type=str, default="vgg16", help="Name of pre-trained model.")
arg = parser.parse_args()


@tf.function
def train_step(_model, _x, _y, _optimizer, _loss, _accuracy):
    with tf.GradientTape() as tape:
        pred = _model(_x, training=True)
        loss = _loss(_y, pred)
        _accuracy.update_state(_y, pred)
    grad = tape.gradient(loss, _model.trainable_variables)
    _optimizer.apply_gradients(zip(grad, _model.trainable_variables))
    return loss, _accuracy.result()

@tf.function
def val_step(_model, _x, _y, _loss, _accuracy):
    pred = _model(_x, training=False)
    loss = _loss(_y, pred)
    _accuracy.update_state(_y, pred)
    return loss, _accuracy.result()


if __name__ == "__main__":

    assert type(arg.epochs)==int, "--epochs must be an integer."
    assert arg.epochs > 1, "--epochs must be bigger than one."
    assert type(arg.name)==str, "--name must be a string."
    assert type(arg.batch)==int, "--batch must be an integer."
    assert arg.batch > 0, "--batch must be bigger than 0."

    model_name = arg.name
    assert not re.fullmatch(r"[A-Za-z_0-9]+\.h5$", model_name) is None, "Invalid model name. It can not finish with .h5."
    model_name += ".h5"

    # Setting up memory use.
    gpus = tf.config.list_physical_devices(device_type="GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

    # Loading data.
    train, val, classes = load_data(arg.batch)

    # Setup metrics.
    train_loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    val_loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    
    train_accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_log = {
        "loss":[],
        "accuracy":[],
        "val_loss":[],
        "val_accuracy":[],
        "epoch":0
    }

    # Loading model to be trained.
    model = VGG16(classes)

    # Calculating progress bar.
    train_bar = len(train) // 10
    val_bar = len(val) // 10

    # Training.
    print("VGG16 model - Starting training..")
    print("|Progress Bar: <==========> 10/10|Batch size: {}|Train Iterations: {}|Val Iterations: {}|".format(arg.batch, len(train),len(val)))
    print("-"*100)
    for e in range(arg.epochs):
        
        print("|Epoch: " + os.path.join(str(e+1),str(arg.epochs)) + "|")
        print("|Train progress:<",end="")
        for steps, (x_train, y_train) in enumerate(train):

            train_loss, train_acc = train_step(model, x_train, y_train, optimizer, train_loss_func, train_accuracy_func)
            if steps % train_bar == 0 and steps != 0:
                print("=",end="")
        print("> - TRAINED",end="|")

        print("Val progress:<",end="")
        for steps, (x_val, y_val) in enumerate(val):

            val_loss, val_acc = val_step(model, x_val, y_val, val_loss_func, val_accuracy_func)
            if steps % val_bar == 0 and steps != 0:
                print("=",end="")
        print("> - VALIDATED|")

        train_log["loss"].append(train_loss)
        train_log["accuracy"].append(train_acc)
        train_log["val_loss"].append(val_loss)
        train_log["val_accuracy"].append(val_acc)
        train_log["epoch"] += 1

        print("|loss: {:.3f} accuracy: {:.3f}|val_loss: {:.3f} val_accuracy: {:.3f}|".format(train_loss, train_acc, val_loss, val_acc))
        print("-"*100)

        train_accuracy_func.reset_states()
        val_accuracy_func.reset_states()

    # Saving model.
    models_path = os.path.dirname(os.path.realpath(__file__)).split("/")
    models_path.pop(-1)
    models_path.append("models")
    models_path = "/".join(models_path)

    model_folder = os.path.join(models_path, "model_" + str(len(os.listdir(models_path)) + 1))
    os.mkdir(model_folder)

    save_train_graph(model_folder, train_log)

    print("Saving model..")
    model.save(os.path.join(model_folder,model_name))
    print("Model saved at " + os.path.join(model_folder,model_name))
    exit(0)
