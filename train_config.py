# TODO: Set the paths to the images.
train_images = '/home/codesteller/datasets/Cityscapes/leftImg8bit/train/'
val_images = '/home/codesteller/datasets/Cityscapes/leftImg8bit/val/'
test_images = '/home/codesteller/datasets/Cityscapes/leftImg8bit/test/'

# TODO: Set the paths to the ground truth images.
train_gt = '/home/codesteller/datasets/Cityscapes/gtFine/train/'
val_gt = '/home/codesteller/datasets/Cityscapes/gtFine/val/'

# TODO: Set pretrained model location
vgg_pretrained = '/home/codesteller/models/pretrained/VGG-16_mod2FCN_ImageNet-Classification'

# Set Training Hyper Parameters
# NOTE: Learning Rate is sen by function learning_rate_schedule in do_train_vgg16fcn.py
num_classes = 34  # TODO: Set the number of segmentation classes.
train_batch_size = 1  # TODO: Set the training batch size.
val_batch_size = 1  # TODO: Set the validation batch size.
epochs = 10  # TODO: Set the number of epochs to train for.

# donot change this
exp_name = "cityscapes_fcn"
VIZ = False
