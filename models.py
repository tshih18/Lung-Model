from keras.models import Model
from keras.layers import Input, average, merge, concatenate,  Reshape, core, Lambda
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, Dropout, ThresholdedReLU, ZeroPadding2D, Cropping2D, Convolution2D
from keras.optimizers import Adam
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from keras.utils import multi_gpu_model, Sequence


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn


def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h / 2, crop_h / 2 + rem_h)
    crop_w_dims = (crop_w / 2, crop_w / 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped


def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)


def get_fcn_model(input_shape, num_classes, weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool1')(mvn3)


    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool2')(mvn7)


    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)


    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)


    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample3')(fuse_scores2)
    crop3 = Lambda(crop, name='crop3')([data, upsample3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=activation, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='predictions')(crop3)

    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef])

    return model

def get_patches_unet5(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = concatenate([conv3,up2], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2,2))(conv7)
    up3 = concatenate([conv2,up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)

    up4 = UpSampling2D(size=(2,2))(conv8)
    up4 = concatenate([conv1,up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv9)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    conv10 = core.Activation('softmax')(conv10)
    #conv10 = ThresholdedReLU(theta=0.25)(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    # multi_model = multi_gpu_model(model, gpus=2)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def get_patches_unet4(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format='channels_first')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format='channels_first')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_first')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_first')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', data_format='channels_first')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', data_format='channels_first')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', data_format='channels_first')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', data_format='channels_first')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Activation('relu')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3,up1], axis=1)
    conv5 = Conv2D(128, (3, 3), padding='same', data_format='channels_first')(up1)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(128, (3, 3), padding='same', data_format='channels_first')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)

    up2 = UpSampling2D(size=(2,2))(conv5)
    up2 = concatenate([conv2,up2], axis=1)
    conv6 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(up2)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(64, (3, 3), padding='same', data_format='channels_first')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Activation('relu')(conv6)

    up3 = UpSampling2D(size=(2,2))(conv6)
    up3 = concatenate([conv1,up3], axis=1)
    conv7 = Conv2D(32, (3, 3), padding='same', data_format='channels_first')(up3)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(32, (3, 3), padding='same', data_format='channels_first')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Activation('relu')(conv7)

    # # Output shape as entire image
    # conv8 = Conv2D(2, (1, 1), padding='same', data_format='channels_first')(conv7)
    # conv8 = BatchNormalization(axis=1)(conv8)
    # conv8 = Activation('relu')(conv8)
    # conv8 = Conv2D(1, (1, 1), padding='same', data_format='channels_first')(conv8)
    # conv8 = BatchNormalization(axis=1)(conv8)
    # conv8 = Dense(1, data_format='channels_first')(conv8)
    # conv8 = core.Activation('softmax')(conv8)

    # Output shape as batches, height*width, 2
    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv7)
    conv8 = core.Reshape((2,patch_height*patch_width))(conv8)
    conv8 = core.Permute((2,1))(conv8)
    conv8 = core.Activation('softmax')(conv8)
    #
    model = Model(inputs=inputs, outputs=conv8)
    # multi_model = multi_gpu_model(model, gpus=2)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='sgd', loss=dice_coef_loss , metrics=['accuracy', dice_coef, jaccard_coef])

    return model

def get_patches_unet3(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format='channels_first')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format='channels_first')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',data_format='channels_first')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',data_format='channels_first')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',data_format='channels_first')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same',data_format='channels_first')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Activation('relu')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',data_format='channels_first')(up1)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',data_format='channels_first')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Activation('relu')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3),padding='same',data_format='channels_first')(up2)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',data_format='channels_first')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Activation('relu')(conv5)
    #
    conv6 = Conv2D(filters=2, kernel_size=(1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    # multi_model = multi_gpu_model(model, gpus=2)
    # model = make_parallel(model, 2)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

    #multi_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_full_unet3(n_ch,img_height,img_width):
    inputs = Input(shape=(n_ch,img_height,img_width))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(filters=1, kernel_size=(1, 1), activation='softmax',padding='same',data_format='channels_first')(conv5)
    # conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    # conv6 = core.Permute((2,1))(conv6)
    ############
    # conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv6)
    # multi_model = multi_gpu_model(model, gpus=2)
    # model = make_parallel(model, 2)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

    #multi_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_full_unet5(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = concatenate([conv3,up2], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2,2))(conv7)
    up3 = concatenate([conv2,up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)

    up4 = UpSampling2D(size=(2,2))(conv8)
    up4 = concatenate([conv1,up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid',padding='same',data_format='channels_first')(conv9)
    # conv10 = core.Activation('softmax')(conv10)
    #conv10 = ThresholdedReLU(theta=0.25)(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    # multi_model = multi_gpu_model(model, gpus=2)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model
