# train_phased.py
import sys
import os
import pandas as pd
sys.path.append('src')

def train_phased():
    """Entrenamiento en dos fases: primero general, luego fine-tuning"""
    from training.trainer import TrainingPipeline
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    
    print("🎯 ENTRENAMIENTO EN DOS FASES")
    print("=" * 50)
    
    # FASE 1: Entrenamiento general
    print("\n🔄 FASE 1: Entrenamiento general")
    pipeline = TrainingPipeline(tls_id="J8")
    pipeline.config = {
        "simulation_runs": 6,
        "steps_per_run": 4000,
        "test_size": 0.15,
        "epochs": 100,  # Menos épocas iniciales
        "model_save_path": "data/trained_models/",
    }
    
    model_path, history, _ = pipeline.run_training(evaluate=False)
    
    # FASE 2: Fine-tuning
    print("\n🎨 FASE 2: Fine-tuning")
    
    # Cargar modelo pre-entrenado
    model = load_model(model_path, compile=False)
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
    )
    
    # Recolectar datos adicionales para fine-tuning
    additional_data = pipeline.collect_training_data()
    df = pd.DataFrame(additional_data)
    X, y, _, _ = pipeline.preprocessor.prepare_training_data(df)
    
    # Fine-tuning con menos épocas
    fine_tune_history = model.fit(
        X, y,
        epochs=50,  # Solo fine-tuning
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Guardar modelo fine-tuned
    fine_tuned_path = model_path.replace('.h5', '_fine_tuned.h5')
    model.save(fine_tuned_path)
    
    print(f"✅ Fine-tuning completado: {fine_tuned_path}")
    return fine_tuned_path

if __name__ == "__main__":
    train_phased()