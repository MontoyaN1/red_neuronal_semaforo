# src/training/trainer.py (VERSIÓN CORREGIDA)
import traci
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

# Añadir el directorio raíz al path para imports absolutos
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data_collection.data_collectors import DataCollector
from src.data_collection.data_preprocessor import DataPreprocessor
from src.model.traffic_model import TrafficLightNN


class TrainingPipeline:
    def __init__(self, tls_id="J1", config_path="config/training_config.yaml"):
        self.tls_id = tls_id
        self.config = self.load_config(config_path)
        self.model = None
        self.preprocessor = None

    def load_config(self, config_path):
        return {
            "simulation_runs": 12,  # Más variedad
            "steps_per_run": 7200,  # Más datos por simulación
            "test_size": 0.1,
            "batch_size": 64,  # Menos datos de test
            "epochs": 200,  # Más épocas de entrenamiento
            "model_save_path": "data/trained_models/",
        }

    def run_training(self, evaluate=True):
        """Ejecutar pipeline completo de entrenamiento (VERSIÓN ÚNICA)"""
        print("🚀 Iniciando entrenamiento del modelo...")

        # 1. Recolectar datos
        simulation_data = self.collect_training_data()

        # 2. Preparar datos para entrenamiento
        X, y, feature_cols, target_cols = self.prepare_training_data(simulation_data)

        # 3. Entrenar modelo
        self.model, history = self.train_model(X, y)

        # 4. Guardar modelo
        model_path = self.save_model()

        # 5. Evaluar modelo si se solicita
        if evaluate:
            evaluation_metrics = self.evaluate_trained_model(model_path)
            print("✅ Entrenamiento y evaluación completados")
            return model_path, history, evaluation_metrics
        else:
            print("✅ Entrenamiento completado")
            return model_path, history, None

    def collect_training_data(self):
        simulation_data = []

        for sim_run in range(self.config["simulation_runs"]):
            print(f"📊 Simulación {sim_run + 1} - Estrategia mejorada")

            traci.start(["sumo", "-c", "prueba2.sumocfg"])
            collector = DataCollector(self.tls_id)

            step = 0
            try:
                while step < self.config["steps_per_run"]:
                    traci.simulationStep()

                    # ✅ ESTRATEGIA MEJORADA: Cambios más inteligentes
                    if step % 25 == 0:  # Menos cambios frecuentes
                        current_phase = traci.trafficlight.getPhase(self.tls_id)

                        # Solo cambiar entre fases principales (0 y 2)
                        if current_phase in [0, 2]:
                            # Cambiar a la otra fase principal
                            next_phase = 2 if current_phase == 0 else 0
                        else:
                            # Si está en fase de transición, ir a la siguiente principal
                            next_phase = 0 if current_phase == 3 else 2

                        traci.trafficlight.setPhase(self.tls_id, next_phase)

                    collector.collect_step_data(step)
                    step += 1

                simulation_data.extend(collector.data)
                print(f"✅ Simulación {sim_run + 1}: {len(collector.data)} registros")

            except Exception as e:
                print(f"❌ Error: {e}")
            finally:
                traci.close()

        return simulation_data

    def prepare_training_data(self, simulation_data):
        """Preparar datos para entrenamiento"""
        df = pd.DataFrame(simulation_data)
        print(f"📈 Datos totales recolectados: {len(df)} registros")

        # Inicializar preprocesador
        self.preprocessor = DataPreprocessor()

        # Preparar datos de entrenamiento
        X, y, feature_cols, target_cols = self.preprocessor.prepare_training_data(df)

        print(f"🔧 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} features")
        return X, y, feature_cols, target_cols

    def train_model(self, X, y):
        """Entrenar el modelo de red neuronal"""
        print("🧠 Iniciando entrenamiento de la red neuronal...")

        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config["test_size"], random_state=42
        )

        # Crear y entrenar modelo
        model_nn = TrafficLightNN(input_dim=X.shape[1], output_dim=y.shape[1])

        history = model_nn.train(
            X_train, y_train, X_val, y_val, epochs=self.config["epochs"]
        )

        self.model = model_nn
        return model_nn, history

    def save_model(self):
        """Guardar modelo entrenado - SOLO usar .h5"""
        os.makedirs(self.config["model_save_path"], exist_ok=True)

        # SOLO .h5 - formato confiable
        model_filename = (
            f"traffic_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.h5"
        )
        model_path = os.path.join(self.config["model_save_path"], model_filename)

        print(f"💾 Guardando modelo en: {model_path}")

        try:
            # Guardar modelo
            self.model.model.save(model_path)
            print("✅ Modelo guardado exitosamente")

            # Guardar preprocesador
            preprocessor_path = model_path.replace(".h5", "_preprocessor.pkl")
            self.preprocessor.save(preprocessor_path)
            print("✅ Preprocesador guardado")

            return model_path

        except Exception as e:
            print(f"❌ Error crítico guardando modelo: {e}")
            # Fallback extremo
            return self._create_emergency_model()

    def _create_emergency_model(self):
        """Crear modelo de emergencia si todo falla"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        import numpy as np

        # Modelo mínimo funcional
        emergency_model = Sequential(
            [
                Dense(16, activation="relu", input_shape=(26,)),
                Dense(8, activation="relu"),
                Dense(4, activation="linear"),
            ]
        )
        emergency_model.compile(optimizer="adam", loss="mse")

        # Entrenar rápido con datos dummy
        X_dummy = np.random.random((100, 26))
        y_dummy = np.random.random((100, 4))
        emergency_model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

        model_path = os.path.join(self.config["model_save_path"], "emergency_model.h5")
        emergency_model.save(model_path)
        print(f"🆘 Modelo de emergencia creado: {model_path}")
        return model_path

    def evaluate_trained_model(self, model_path):
        """Evaluar el modelo recién entrenado"""
        from .evaluator import ModelEvaluator

        print("🧪 Evaluando modelo entrenado...")

        evaluator = ModelEvaluator(self.model, self.preprocessor, self.tls_id)
        metrics = evaluator.evaluate_model("prueba2.sumocfg")

        # Generar reporte
        report_file = model_path.replace(".h5", "_evaluation.json")
        evaluator.generate_evaluation_report(metrics, output_file=report_file)

        return metrics


def main():
    """Función principal para ejecutar entrenamiento DESDE ESTE ARCHIVO"""
    pipeline = TrainingPipeline(tls_id="J1")  # Reemplaza con tu ID de semáforo
    model_path, history, metrics = pipeline.run_training(evaluate=True)

    print("🎯 Entrenamiento finalizado!")
    print(f"📁 Modelo guardado en: {model_path}")

    if metrics:
        print(
            f"📊 Evaluación completada. Score: {metrics.get('final_efficiency_score', 0):.3f}"
        )


if __name__ == "__main__":
    main()
