from functionality.dataset_load import *
from functionality.models import *
from functionality.image_augmentation import *

if __name__ == '__main__':
    NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
    NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
    print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES,
                                                                                               NUM_VALIDATION_IMAGES,
                                                                                               NUM_TEST_IMAGES))

    ds_train = get_training_dataset(True)
    ds_valid = get_validation_dataset(True)
    ds_test = get_test_dataset(True)

    # while True:
    #     # show batch of images
    #     ds_iter = iter(ds_train.unbatch().batch(20))
    #     one_batch = next(ds_iter)
    #     display_batch_of_images(one_batch)

    # Define training epochs
    EPOCHS = 50
    STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

    model = vgg16()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    model.summary()

    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        # callbacks=[lr_callback],
    )

    display_training_curves(
        history.history['loss'],
        history.history['val_loss'],
        'loss',
        211,
    )
    display_training_curves(
        history.history['sparse_categorical_accuracy'],
        history.history['val_sparse_categorical_accuracy'],
        'accuracy',
        212,
    )
