# real_time_adaptation.py
import sys
import os
sys.path.append('src')

class RealTimeAdaptiveController:
    def __init__(self, base_model_path, tls_id="J8", adaptation_rate=0.1):
        from src.deployment.traffic_controller import RealTimeController
        from tensorflow.keras.models import load_model
        from src.data_collection.data_preprocessor import DataPreprocessor
        
        self.tls_id = tls_id
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        
        # Cargar modelo base
        self.model = load_model(base_model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load(base_model_path.replace('.h5', '_preprocessor.pkl'))
        
        self.controller = RealTimeController(
            model=self.model, 
            preprocessor=self.preprocessor, 
            tls_id=tls_id
        )
    
    def detect_traffic_pattern(self, state):
        """Detectar el patrón de tráfico actual"""
        total_volume = sum(state.get(f'lane_{i}_volume', 0) for i in range(8))
        max_waiting = max(state.get(f'lane_{i}_waiting', 0) for i in range(8))
        
        # Calcular desbalance
        phase0_volume = sum(state.get(f'lane_{i}_volume', 0) for i in [2, 3, 6, 7])
        phase2_volume = sum(state.get(f'lane_{i}_volume', 0) for i in [0, 1, 4, 5])
        balance_ratio = min(phase0_volume, phase2_volume) / (max(phase0_volume, phase2_volume) + 0.001)
        
        if total_volume < 20:
            return "low_traffic"
        elif max_waiting > 15:
            return "congested" 
        elif balance_ratio < 0.3:
            return "unbalanced"
        else:
            return "normal"
    
    def adaptive_decision(self, state):
        """Decisión adaptativa basada en patrón detectado"""
        pattern = self.detect_traffic_pattern(state)
        base_decision = self.controller.make_decision(state)
        
        # Ajustes basados en patrón
        if pattern == "low_traffic":
            # Ciclos más cortos y balanceados
            adjusted = [max(15, d * 0.8) for d in base_decision]
        elif pattern == "congested":
            # Más tiempo para descongestionar
            adjusted = [min(50, d * 1.2) for d in base_decision]
        elif pattern == "unbalanced":
            # Ajustar balance
            phase0_traffic = sum(state.get(f'lane_{i}_volume', 0) for i in [2, 3, 6, 7])
            phase2_traffic = sum(state.get(f'lane_{i}_volume', 0) for i in [0, 1, 4, 5])
            
            if phase0_traffic > phase2_traffic * 2:
                adjusted = [base_decision[0] * 1.3, 3, base_decision[2] * 0.7, 3]
            else:
                adjusted = [base_decision[0] * 0.7, 3, base_decision[2] * 1.3, 3]
        else:
            adjusted = base_decision
        
        print(f"🎯 Patrón: {pattern} → Ajuste: {[f'{d:.1f}' for d in adjusted]}")
        return adjusted
    
    def run_adaptive_control(self, duration=3600):
        """Ejecutar control adaptativo"""
        import traci
        from src.data_collection.data_collectors import DataCollector
        
        traci.start(["sumo-gui", "-c", "prueba2.sumocfg"])
        collector = DataCollector(self.tls_id)
        
        step = 0
        pattern_counts = {"low_traffic": 0, "normal": 0, "congested": 0, "unbalanced": 0}
        
        while step < duration:
            traci.simulationStep()
            
            if step % self.controller.phase_change_interval == 0:
                state = self.controller.get_current_state()
                decision = self.adaptive_decision(state)
                self.controller.apply_decision(decision)
                
                # Contar patrones
                pattern = self.detect_traffic_pattern(state)
                pattern_counts[pattern] += 1
            
            if step % 300 == 0:  # Cada 5 minutos
                print(f"⏱️  Step {step}: Patrones {pattern_counts}")
            
            step += 1
        
        traci.close()
        
        print(f"\n📊 Estadísticas de patrones:")
        for pattern, count in pattern_counts.items():
            percentage = (count / (duration / self.controller.phase_change_interval)) * 100
            print(f"   {pattern}: {count} veces ({percentage:.1f}%)")

def main():
    base_model = "data/trained_models/traffic_model_20251006_1755.h5"
    adaptive_controller = RealTimeAdaptiveController(base_model)
    
    print("🚀 INICIANDO CONTROL ADAPTATIVO EN TIEMPO REAL")
    adaptive_controller.run_adaptive_control(duration=1800)  # 30 minutos

if __name__ == "__main__":
    main()