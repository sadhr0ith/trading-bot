from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.logger import setup_logger

class LSTMModel:
    def __init__(self, input_shape, units=50, learning_rate=0.001, optimizer_type='adam', use_lr_decay=False):
        self.logger = setup_logger(self.__class__.__name__)
        self.model = Sequential()

        # Define the LSTM model architecture
        self.model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(units, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        # Choose optimizer
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam' or 'rmsprop'.")

        # Compile the model
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.log_action(f"Initialized LSTM model with {units} units, optimizer: {optimizer_type}, lr: {learning_rate}")

        # Use learning rate decay if specified
        self.use_lr_decay = use_lr_decay
        self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1) if use_lr_decay else None

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        # Forward fit method to the internal Keras model
        self.log_action("Training the LSTM model...")
        callbacks = [self.reduce_lr] if self.use_lr_decay else []
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, X_test):
        # Forward predict method to the internal Keras model
        self.log_action("Making predictions with the LSTM model...")
        return self.model.predict(X_test)

    def log_action(self, message):
        self.logger.info(message)

# Function for creating LSTM model for hyperparameter tuning
def create_lstm_model(input_shape, units=50, learning_rate=0.001, optimizer_type='adam'):
    model = Sequential()
    
    # Use the Input layer to specify the input shape
    model.add(Input(shape=input_shape))
    
    # LSTM layer
    model.add(LSTM(units=units, return_sequences=False))
    
    # Output layer
    model.add(Dense(1))
    
    # Select the optimizer
    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model