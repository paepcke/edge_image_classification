import tensorflow as tf
import tensorflow.keras.applications as model_zoo

def build_resnet50(pretrained=True):
    if pretrained:
        weights = 'imagenet'
    else:
        weights = None
    model = model_zoo.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224, 3), classes=3)
    if pretrained:
        model.save('saved_keras_resnet.h5')
    return model

model = build_resnet50()

