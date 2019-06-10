from tensorflow.python.keras import backend as K

def single_class_accuracy(interesting_class_id):
    if interesting_class_id == 0:
        def coyote_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return coyote_acc
    if interesting_class_id == 1:
        def human_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return human_acc
    if interesting_class_id == 2:
        def lion_acc(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return lion_acc


def mean_per_class_accuracy(y_true, y_pred):
  class_id_true = K.argmax(y_true, axis=-1)
  class_id_preds = K.argmax(y_pred, axis=-1)
  # Replace class_id_preds with class_id_true for recall here
  interesting_class_id = 0
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class0_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  interesting_class_id = 1
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class1_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  interesting_class_id = 2
  accuracy_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
  class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
  class2_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)

  return (class0_acc + class1_acc + class2_acc) / 3
