# src/data_collection/data_preprocessor.py
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = None
        self.expected_feature_count = None

    def save(self, filepath):
        """Guardar preprocesador - INCLUIR expected_feature_count"""
        # ✅ Asegurar que expected_feature_count esté configurado
        if self.feature_columns and not hasattr(self, "expected_feature_count"):
            self.expected_feature_count = len(self.feature_columns)

        data = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "expected_feature_count": getattr(self, "expected_feature_count", None),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Preprocesador guardado en: {filepath}")

    def load(self, filepath):
        """Cargar preprocesador guardado"""
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.scaler = data["scaler"]
                self.feature_columns = data["feature_columns"]
                self.target_columns = data["target_columns"]
                # ✅ AÑADIR esta línea:
                self.expected_feature_count = (
                    len(self.feature_columns) if self.feature_columns else None
                )
            print(f"📦 Preprocesador cargado desde: {filepath}")
            print(f"🔍 Features esperadas: {self.expected_feature_count}")
        except Exception as e:
            print(f"❌ Error cargando preprocesador: {e}")

    def prepare_training_data(self, df):
        """Preparar datos para entrenamiento"""
        # Identificar columnas de features automáticamente
        self.feature_columns = [
            col
            for col in df.columns
            if any(
                x in col
                for x in [
                    "lane_",
                    "volume",
                    "waiting",
                    "occupancy",
                    "current_phase",
                    "phase_duration",
                ]
            )
            and col
            not in [
                "total_waiting_time",
                "average_speed",
                "throughput",
                "efficiency_score",
                "simulation_step",
                "simulation_time",
            ]
        ]

        self.expected_feature_count = len(self.feature_columns)  # ⬅️ GUARDAR
        print(
            f"🔧 Número de features para entrenamiento: {self.expected_feature_count}"
        )

        X = df[self.feature_columns].fillna(0).values
        y = self._calculate_optimal_durations(df)

        # Escalar features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, self.feature_columns, self.target_columns

    def prepare_inference_data(self, state_dict):
        """Preparar datos para inferencia (evaluación/tiempo real)"""
        if self.feature_columns is None:
            raise ValueError(
                "Preprocesador no entrenado. Ejecuta prepare_training_data primero."
            )

        # Crear array con las features en el ORDEN CORRECTO
        features = []
        for col in self.feature_columns:
            features.append(state_dict.get(col, 0.0))  # Usar 0 si falta alguna feature

        features_array = np.array([features])

        # Verificar dimensión
        if len(features) != self.expected_feature_count:
            print(
                f"⚠️  Advertencia: Features esperadas {self.expected_feature_count}, obtenidas {len(features)}"
            )

        # Escalar
        features_scaled = self.scaler.transform(features_array)
        return features_scaled

    def _calculate_optimal_durations(self, df):
        optimal_durations = []
        
        for _, row in df.iterrows():
            # Calcular por fase con mapeo corregido
            phase_volumes = [0, 0]
            phase_waiting = [0, 0]
            
            for lane_idx in [2, 3, 6, 7]:  # Fase 0
                phase_volumes[0] += row.get(f'lane_{lane_idx}_volume', 0)
                phase_waiting[0] += row.get(f'lane_{lane_idx}_waiting', 0)
            
            for lane_idx in [0, 1, 4, 5]:  # Fase 2
                phase_volumes[1] += row.get(f'lane_{lane_idx}_volume', 0)
                phase_waiting[1] += row.get(f'lane_{lane_idx}_waiting', 0)
            
            # ✅ ESTRATEGIA MEJORADA: Considerar congestión severa
            max_waiting = max(phase_waiting)
            
            if max_waiting > 10:
                # Congestión severa - dar más tiempo a fases con colas largas
                base_cycle = 70
                weights = [w * 4 + v for v, w in zip(phase_volumes, phase_waiting)]
            else:
                # Tráfico normal - balance entre volumen y espera
                base_cycle = 60
                weights = [w * 2 + v for v, w in zip(phase_volumes, phase_waiting)]
            
            total_weight = sum(weights) if sum(weights) > 0 else 1
            main_durations = [int(base_cycle * (w / total_weight)) for w in weights]
            
            # Asegurar límites razonables
            main_durations = [max(15, min(55, d)) for d in main_durations]
            
            optimal_durations.append([main_durations[0], 3, main_durations[1], 3])
        
        return np.array(optimal_durations)