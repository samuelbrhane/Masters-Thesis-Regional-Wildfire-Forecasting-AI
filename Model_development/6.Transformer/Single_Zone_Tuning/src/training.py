import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from constants import EARLY_STOPPING_PATIENCE, MAX_EPOCHS, LOSS_FUNCTION

def train_model(model, X_train, y_train, X_val, y_val, params):
    """
    Compile and train the Transformer model with multiple inputs.
    """

    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {params['optimizer']}")


    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history
