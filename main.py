import argparse
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.training.trainer import TrainingPipeline
from src.training.evaluator import ModelEvaluator


def load_preprocessor(preprocessor_path):
    """Cargar preprocesador desde archivo"""
    try:
        with open(preprocessor_path, "rb") as f:
            preprocessor_data = pickle.load(f)

        from src.data_collection.data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        preprocessor.scaler = preprocessor_data["scaler"]
        preprocessor.feature_columns = preprocessor_data["feature_columns"]
        preprocessor.target_columns = preprocessor_data["target_columns"]

        print(f"✅ Preprocesador cargado: {preprocessor_path}")
        return preprocessor

    except FileNotFoundError:
        print(f"❌ No se encontró el preprocesador: {preprocessor_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Sistema de Semáforo Inteligente")
    parser.add_argument("--mode", choices=["train", "eval", "run"], required=True)
    parser.add_argument("--model", help="Ruta al modelo para eval/run")
    parser.add_argument("--tls", default="J1", help="ID del semáforo")
    parser.add_argument(
        "--scenario", default="prueba2.sumocfg", help="Escenario de prueba"
    )
    parser.add_argument("--gui", action="store_true", help="Usar interfaz gráfica")

    args = parser.parse_args()

    if args.mode == "train":
        # Entrenar y evaluar automáticamente
        print("🎯 Modo: ENTRENAMIENTO")
        pipeline = TrainingPipeline(tls_id=args.tls)
        model_path, history, metrics = pipeline.run_training(evaluate=True)

        print(f"✅ Entrenamiento completado. Modelo: {model_path}")

    elif args.mode == "eval":
        # Solo evaluación
        print("🎯 Modo: EVALUACIÓN")
        if not args.model:
            print("❌ Se requiere --model para evaluación")
            return

        # Cargar modelo - CON COMPILE=FALSE
        from tensorflow.keras.models import load_model  # type: ignore

        try:
            model = load_model(args.model, compile=False)
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            print("✅ Modelo cargado y recompilado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return

        # Cargar preprocesador
        preprocessor_path = args.model.replace(".h5", "_preprocessor.pkl")
        preprocessor = load_preprocessor(preprocessor_path)

        if preprocessor is None:
            print("❌ No se puede evaluar sin preprocesador")
            return

        # Evaluar
        evaluator = ModelEvaluator(model, preprocessor, args.tls)
        metrics = evaluator.evaluate_model(args.scenario)
        evaluator.generate_evaluation_report(metrics)

    elif args.mode == "run":
        # Ejecución en tiempo real
        print("🎯 Modo: EJECUCIÓN EN TIEMPO REAL")
        if not args.model:
            print("❌ Se requiere --model para modo run")
            return

        # Importar aquí para evitar dependencias circulares
        from .src.deployment.traffic_controller import RealTimeController

        controller = RealTimeController(
            model_path=args.model, tls_id=args.tls, use_gui=args.gui
        )
        controller.run()


if __name__ == "__main__":
    main()
