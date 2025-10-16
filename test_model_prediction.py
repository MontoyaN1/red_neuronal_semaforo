# test_model_prediction_fixed.py
import sys
import os
sys.path.append('src')

def test_model_prediction():
    from tensorflow.keras.models import load_model
    import numpy as np
    
    # Buscar el modelo más reciente
    model_dir = "data/trained_models"
    if not os.path.exists(model_dir):
        print(f"❌ No existe la carpeta: {model_dir}")
        return
    
    # Buscar archivos de modelo
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.h5') or file.endswith('.keras'):
            model_files.append(file)
    
    if not model_files:
        print("❌ No se encontraron archivos de modelo")
        return
    
    # Usar el más reciente
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    print(f"🔍 Probando modelo: {latest_model}")
    
    try:
        # Cargar modelo
        model = load_model(model_path)
        print(f"✅ Modelo cargado: {model_path}")
        
        # Probar predicción
        sample_input = np.random.random((1, 26))
        
        # Método 1: predict
        pred1 = model.predict(sample_input, verbose=0)
        print(f"✅ Predict: {pred1[0]}")
        
        # Método 2: __call__
        pred2 = model(sample_input, training=False)
        print(f"✅ Call: {pred2[0].numpy()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_model_prediction()