# potential_test.py
import sys
import os
sys.path.append('src')

def test_potential():
    """Verificar si el modelo PUEDE superar al tradicional"""
    from src.deployment.traffic_controller import RealTimeController
    from tensorflow.keras.models import load_model
    from src.data_collection.data_preprocessor import DataPreprocessor
    import traci
    
    print("🎯 TEST DE POTENCIAL: ¿Puede el modelo ser MEJOR?")
    
    # Cargar modelo actual
    model_path = "data/trained_models/traffic_model_20251006_1650.h5"
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    preprocessor = DataPreprocessor()
    preprocessor.load(model_path.replace('.h5', '_preprocessor.pkl'))
    
    # Probar en escenario específico
    controller = RealTimeController(model=model, preprocessor=preprocessor, tls_id="J8")
    
    # Ejecutar y monitorear decisiones
    traci.start(["sumo-gui", "-c", "prueba2.sumocfg"])
    
    step = 0
    good_decisions = 0
    total_decisions = 0
    
    while step < 300:
        traci.simulationStep()
        
        if step % 10 == 0:
            state = controller.get_current_state()
            decision = controller.make_decision(state)
            
            # Evaluar si la decisión es "inteligente"
            total_volume = sum(state.get(f'lane_{i}_volume', 0) for i in range(8))
            max_waiting = max(state.get(f'lane_{i}_waiting', 0) for i in range(8))
            
            # Una decisión es "buena" si da más tiempo donde hay más tráfico
            phase0_traffic = sum(state.get(f'lane_{i}_volume', 0) for i in [2, 3, 6, 7])
            phase2_traffic = sum(state.get(f'lane_{i}_volume', 0) for i in [0, 1, 4, 5])
            
            if (phase0_traffic > phase2_traffic and decision[0] > decision[2]) or \
               (phase2_traffic > phase0_traffic and decision[2] > decision[0]):
                good_decisions += 1
            
            total_decisions += 1
            
            print(f"Step {step}: F0={phase0_traffic}, F2={phase2_traffic} → {decision}")
        
        step += 1
    
    traci.close()
    
    accuracy = good_decisions / total_decisions if total_decisions > 0 else 0
    print(f"\n🎯 Precisión de decisiones: {accuracy:.1%}")
    
    if accuracy > 0.7:
        print("✅ El modelo TIENE POTENCIAL para superar al tradicional")
    else:
        print("❌ El modelo necesita más refinamiento")

if __name__ == "__main__":
    test_potential()