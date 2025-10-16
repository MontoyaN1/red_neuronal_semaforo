from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np


class TrafficLightNN:
    def __init__(self, input_dim, output_dim, learning_rate=0.0003):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        """Construir la arquitectura de la red neuronal"""
        # volumen de vehículos, ocupación, tiempo espera, fase actual
        inputs = Input(shape=(self.input_dim,))

       # Arquitectura ligeramente más profunda
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # Duración de cada fase
        outputs = Dense(self.output_dim, activation="softplus")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",  # Mean Squared Error
            metrics=["mae"],  # Mean Absolute Error
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Entrenar el modelo"""
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1,
        )
        return history

    def predict(self, X, verbose=0):
        """Método predict compatible con Keras"""
        return self.model.predict(X, verbose=verbose)

    def predict_phase_durations(self, current_state):
        """Método existente - mantener por compatibilidad"""
        predictions = self.model.predict(np.array([current_state]), verbose=0)
        return predictions[0]
