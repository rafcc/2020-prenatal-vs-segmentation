import os 
import datetime
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.layers import Add
from keras.utils import plot_model
from nn.unet import UNet
from nn.autoencoder import AutoEncoder
from util.data import load_X, load_Xseq, load_Xvgg, load_Y


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def train_unet(inpdir, tardir, val_inpdir, val_tardir, modelpath, batch, total_epoch, save_epoch, size):
    # config
    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = 64

    # Data load
    X_train, file_names = load_X(inpdir, size=size)
    Y_train             = load_Y(tardir, size=size)
    X_test, _ = load_X(val_inpdir, size=size)
    Y_test    = load_Y(val_tardir, size=size)

    # U-Net
    network = UNet(input_channel_count, output_channel_count, size, first_layer_filter_count)
    model = network.create_model()
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=[dice_coef])
    plot_model(model, to_file=os.path.dirname(modelpath)+"/unet.png", show_shapes=True)

    # set checkpoint
    model_checkpoint = ModelCheckpoint(modelpath.rsplit(".",1)[0]+".{epoch:03d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False , save_weights_only=True, period=save_epoch)
    model_checkpoint_best = ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # set logfile
    now = datetime.datetime.now()
    csv_logger = CSVLogger(os.path.dirname(modelpath)+"/log_unet_"+now.strftime('%Y%m%d_%H%M%S')+".csv")

    # train
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch, epochs=total_epoch, verbose=1, callbacks=[model_checkpoint, model_checkpoint_best, csv_logger])


def train_ae_unet(inpdir, tardir, vggdir, val_inpdir, val_tardir, val_vggdir, modelpath, batch, total_epoch, save_epoch, num, size, fine_tuning=False, pretrained_modelpath=None,VGG_flag=True):
    # config
    unet_inch  = 1
    unet_outch = 1
    ae_inch    = int(num*2+1)
    ae_outch   = 1
    first_layer_filter_count = 64

    # Data load
    X_train, file_names = load_X(inpdir, size)
    Xseq_train = load_Xseq(inpdir, num, size)
    Xvgg_train = load_Xvgg(vggdir)
    Y_train    = load_Y(tardir, size=size)
    X_test, _  = load_X(val_inpdir, size)
    Xseq_test  = load_Xseq(val_inpdir, num, size)
    Xvgg_test  = load_Xvgg(val_vggdir)
    Y_test     = load_Y(val_tardir, size=size)

    # U-Net
    UNET = UNet(unet_inch, unet_outch, size, first_layer_filter_count)
    unet = UNET.create_model()

    # AutoEncoder
    AE = AutoEncoder(ae_inch, ae_outch, size, first_layer_filter_count)
    ae = AE.create_model(vgg_on=VGG_flag, fine_tuning=fine_tuning, pretrained_modelpath=pretrained_modelpath)
 
    # U-Net + AutoEncoder
    added = Add()([unet.output, ae.output])
    model = Model(inputs = [unet.input, ae.input[0], ae.input[1]], outputs = added) 
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=[dice_coef])
    plot_model(model, to_file=os.path.dirname(modelpath)+"/ae_unet.png", show_shapes=True)

    # set checkpoint
    model_checkpoint = ModelCheckpoint(modelpath.rsplit(".",1)[0]+".{epoch:03d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False , save_weights_only=True, period=save_epoch)
    model_checkpoint_best = ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

    # set logfile
    now = datetime.datetime.now()
    csv_logger = CSVLogger(os.path.dirname(modelpath)+"/log_ae_unet_"+now.strftime('%Y%m%d_%H%M%S')+".csv")

    # train
    history = model.fit([X_train, Xseq_train, Xvgg_train], Y_train, validation_data=([X_test, Xseq_test, Xvgg_test],Y_test), batch_size=batch, epochs=total_epoch, verbose=1, callbacks=[model_checkpoint, model_checkpoint_best, csv_logger])


def train_ae(inpdir, val_inpdir, modelpath, batch, total_epoch, save_epoch, num, size):
    '''
    train AutoEncoder only
    '''
    # config
    input_channel_count = 7
    output_channel_count = 7
    first_layer_filter_count = 64

    # Data load
    Xseq_train = load_Xseq(inpdir,     num=num, size=size)
    Xseq_test  = load_Xseq(val_inpdir, num=num, size=size)

    # AutoEncoder
    AE = AutoEncoder(input_channel_count, output_channel_count, size, first_layer_filter_count)
    model = AE.create_model(vgg_on=False)
    model.compile(loss="binary_crossentropy", optimizer=Adam())
    plot_model(model, to_file=os.path.dirname(modelpath)+"/ae.png", show_shapes=True)

    # set checkpoint
    model_checkpoint = ModelCheckpoint(modelpath.rsplit(".",1)[0]+".{epoch:03d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, period=save_epoch)
    model_checkpoint_best = ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

    # set logfile
    now = datetime.datetime.now()
    csv_logger = CSVLogger(os.path.dirname(modelpath)+"/log_ae"+now.strftime('%Y%m%d_%H%M%S')+".csv")

    history = model.fit(Xseq_train, Xseq_train, validation_data=(Xseq_test, Xseq_test),batch_size=batch, epochs=total_epoch, verbose=1, callbacks=[model_checkpoint, model_checkpoint_best, csv_logger])