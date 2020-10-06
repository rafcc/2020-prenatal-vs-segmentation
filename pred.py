import os
import cv2
from tqdm import tqdm
from keras.models import Model
from keras.layers import Add
from nn.unet import UNet
from nn.autoencoder import AutoEncoder
from util.data import denormalize_y
from util.data import load_X, load_Xseq, load_Xvgg, load_Y

def pred_Unet(inpdir, outdir, modelpath, batch, size):
    # config
    input_channel_count = 1
    output_channel_count = 1
    first_layer_filter_count = 64

    # network load
    network = UNet(input_channel_count, output_channel_count, size, first_layer_filter_count)
    model = network.create_model()
    model.load_weights(modelpath)

    # data load
    X_test, file_names = load_X(inpdir, size)

    # predict
    Y_pred = model.predict(X_test, batch)

    # output predict image
    for i, y in enumerate(tqdm(Y_pred)):
        img = cv2.imread(os.path.join(inpdir, file_names[i]))
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(outdir, file_names[i]), denormalize_y(y))

def pred_Proposed(inpdir, vggdir, outdir, modelpath, batch, num, size,VGG_flag=True):
    # config
    unet_inch  = 1
    unet_outch = 1
    ae_inch    = int(num*2+1)
    ae_outch   = 1
    first_layer_filter_count = 64

    # dataload
    X_test, file_names = load_X(inpdir, size)
    Xseq_train = load_Xseq(inpdir, num, size)
    Xvgg_train = load_Xvgg(vggdir)

    # U-Net
    UNET = UNet(unet_inch, unet_outch, size, first_layer_filter_count)
    unet = UNET.create_model()

    # AutoEncoder
    AE = AutoEncoder(ae_inch, ae_outch, size, first_layer_filter_count)
    ae = AE.create_model(vgg_on=VGG_flag)
 
    # U-Net + AutoEncoder
    added = Add()([unet.output, ae.output])
    model = Model(inputs = [unet.input, ae.input[0], ae.input[1]], outputs = added) 
    model.load_weights(modelpath)
 
    # predict
    Y_pred = model.predict([X_test, Xseq_train, Xvgg_train], batch)

    # output predict image
    for i, y in enumerate(tqdm(Y_pred)):
        img = cv2.imread(os.path.join(inpdir, file_names[i]))
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(outdir, file_names[i]), denormalize_y(y))
