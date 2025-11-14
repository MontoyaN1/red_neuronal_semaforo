import traci
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data_collection.data_collectors import DataCollector


class RealTimeController:
    def __init__(
        self, model_path=None, model=None, preprocessor=None, tls_id="J1", use_gui=True
    ):
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.phase_change_interval = 10

        # Cargar modelo de diferentes maneras
        if model is not None:
            # Si ya nos pasan el modelo cargado
            self.model = model
            self.preprocessor = preprocessor
        elif model_path is not None:
            # Si nos pasan una ruta, cargarlo
            self.model = self.load_model(model_path)
            self.preprocessor = self.load_preprocessor(model_path)
        else:
            raise ValueError("Se requiere model_path o model")

        self.collector = DataCollector(tls_id)

    def load_model(self, model_path):
        """Cargar modelo de manera compatible"""
        print(f"📦 Cargando modelo desde: {model_path}")

        if not os.path.exists(model_path):
            print(f"❌ Archivo no existe: {model_path}")
            return None

        try:
            from tensorflow.keras.models import load_model  # type: ignore

            # ✅ USAR compile=False para evitar problemas de compatibilidad
            model = load_model(model_path, compile=False)

            # ✅ Recompilar manualmente
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            print("✅ Modelo cargado y recompilado exitosamente")
            return model

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None

    def load_preprocessor(self):
        """Cargar preprocesador"""
        preprocessor_path = self.model_path.replace(".h5", "_preprocessor.pkl")
        print(f"📦 Cargando preprocesador desde: {preprocessor_path}")

        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                return pickle.load(f)
        else:
            print("⚠️  No se encontró preprocesador, usando uno por defecto")
            from src.data_collection.data_preprocessor import DataPreprocessor

            return DataPreprocessor()

    def count_waiting_vehicles(self, lane):
        """De TrafficLightAgent - contar vehículos detenidos"""
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        waiting = 0
        for veh_id in vehicles:
            if traci.vehicle.getSpeed(veh_id) < 0.1:
                waiting += 1
        return waiting

    def get_current_state(self, step=None):
        """Combinación de ambos métodos - obtener estado actual"""
        state = {}
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        for i, lane in enumerate(lanes):
            state[f"lane_{i}_volume"] = len(traci.lane.getLastStepVehicleIDs(lane))
            state[f"lane_{i}_occupancy"] = traci.lane.getLastStepOccupancy(lane)
            state[f"lane_{i}_waiting"] = self.count_waiting_vehicles(lane)

        state["current_phase"] = traci.trafficlight.getPhase(self.tls_id)
        state["phase_duration"] = traci.trafficlight.getPhaseDuration(self.tls_id)

        return state

    def make_decision(self, current_state):
        """Tomar decisión basada en el estado actual"""
        if self.preprocessor is None or self.model is None:
            print("⚠️  No hay preprocesador o modelo, usando valores por defecto")
            return [30, 30, 30, 30]

        try:
            # Preprocesar
            state_scaled = self.preprocessor.prepare_inference_data(current_state)

            # LLAMADA DIRECTA al modelo de Keras
            predictions = self.model.predict(state_scaled, verbose=0)
            return predictions[0]
        except Exception as e:
            print(f"❌ Error en make_decision: {e}")
            # Debug: mostrar qué está pasando
            print(f"🔍 Estado recibido: {current_state}")
            print(f"🔍 Tipo de modelo: {type(self.model)}")
            return [30, 30, 30, 30]

    def apply_decision(self, phase_durations):
        """Aplicar la decisión al semáforo"""
        # Configurar nuevas duraciones para cada fase
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]

        for i, phase in enumerate(logic.phases):
            if i < len(phase_durations):
                # Actualizar duración de la fase con límites
                phase.duration = max(5, min(60, phase_durations[i]))

        # Aplicar nueva lógica al semáforo
        traci.trafficlight.setProgramLogic(self.tls_id, logic)
        print(
            f"🔄 Semáforo actualizado. Duraciones: {[f'{d:.1f}s' for d in phase_durations]}"
        )

    def run(self):
        """Ejecutar control en tiempo real"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        print("🚦 Iniciando control inteligente en tiempo real...")
        traci.start([sumo_binary, "-c", "prueba2.sumocfg"])

        step = 0

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()

                # Tomar decisión cada cierto intervalo
                if step % self.phase_change_interval == 0:
                    state = self.get_current_state(step)
                    decisions = self.make_decision(state)
                    self.apply_decision(decisions)

                # Mostrar progreso
                if step % 100 == 0:
                    vehicles = traci.vehicle.getIDList()
                    print(f"⏱️  Step {step}: {len(vehicles)} vehículos activos")

                step += 1

        except KeyboardInterrupt:
            print("🛑 Simulación detenida por el usuario")
        except Exception as e:
            print(f"❌ Error durante ejecución: {e}")
        finally:
            traci.close()
            print("✅ Control finalizado")
