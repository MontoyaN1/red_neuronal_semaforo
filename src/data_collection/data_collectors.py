import traci
import pandas as pd


class DataCollector:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.data = []
        self.lanes = []

    def get_throughput(self, lanes=None):
        """Calcular vehiculos presentes en los carriles"""
        if lanes is None:
            lanes = self.lanes
        total_vehicles = 0
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            total_vehicles += len(vehicles)
        return total_vehicles  # densidad instantenea

    def get_max_queue_length(self, lanes=None):
        """Obtener longitud máxima de cola"""
        if lanes is None:
            lanes = self.lanes
        max_queue = 0
        for lane in lanes:
            queue_length = self.count_waiting_vehicles(lane)
            if queue_length > max_queue:
                max_queue = queue_length
        return max_queue

    def get_emergency_vehicles(self, lanes=None):
        """Contar vehículos de emergencia"""
        if lanes is None:
            lanes = self.lanes
        emergency_count = 0
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicles:
                vehicle_type = traci.vehicle.getTypeID(veh_id)
                if "emergency" in vehicle_type.lower():
                    emergency_count += 1
        return emergency_count

    def collect_step_data(self, step):
        """Recolectar datos COMPLETOS de cada step"""
        # Obtener lanes controlados si no los tenemos
        if not self.lanes:
            self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        features = {}

        # 1. Datos por carril
        for i, lane in enumerate(self.lanes):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            vehicle_count = len(vehicle_ids)

            waiting_vehicles = self.count_waiting_vehicles(lane)

            features[f"lane_{i}_volume"] = vehicle_count
            features[f"lane_{i}_waiting"] = waiting_vehicles
            features[f"lane_{i}_occupancy"] = traci.lane.getLastStepOccupancy(lane)

        # 2. Estado del semáforo
        features["current_phase"] = traci.trafficlight.getPhase(self.tls_id)
        features["phase_duration"] = traci.trafficlight.getPhaseDuration(self.tls_id)

        # 3. Métricas de desempeño GLOBALES
        features["total_waiting_time"] = self.get_total_waiting_time(self.lanes)
        features["average_speed"] = self.get_average_speed(self.lanes)
        features["throughput"] = self.get_throughput(self.lanes)
        features["max_queue_length"] = self.get_max_queue_length(self.lanes)

        # 4. Información temporal
        features["simulation_step"] = step
        features["simulation_time"] = traci.simulation.getTime()

        self.data.append(features)
        return features

    # Vehiculos que esperan
    def count_waiting_vehicles(self, lane):
        """Contar vehículos detenidos (velocidad < 0.1 m/s)"""
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        waiting = 0
        for veh in vehicles:
            if traci.vehicle.getSpeed(veh) < 0.1:
                waiting += 1
        return waiting

    def get_total_waiting_time(self, lanes):
        """Calcular tiempo total de espera"""
        total_waiting = 0
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                total_waiting += traci.vehicle.getWaitingTime(veh)
        return total_waiting

    def get_average_speed(self, lanes):
        """Calcular velocidad promedio de vehículos en los lanes controlados"""
        total_speed = 0.0
        vehicle_count = 0

        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicles:
                speed = traci.vehicle.getSpeed(veh_id)
                total_speed += speed
                vehicle_count += 1

        # Evitar división por cero
        if vehicle_count > 0:
            return total_speed / vehicle_count
        else:
            return 0.0

    def save_data(self, filename):
        """Guardar datos recolectados"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
