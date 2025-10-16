# src/training/evaluator.py
import traci
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Añadir path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data_collection.data_collectors import DataCollector
from src.deployment.traffic_controller import RealTimeController


class ModelEvaluator:
    def __init__(self, model, preprocessor, tls_id="J1"):
        self.model = model
        self.preprocessor = preprocessor
        self.tls_id = tls_id
        self.metrics_history = []

    def evaluate_model(self, test_scenario, evaluation_duration=1800):
        """Evaluar el modelo en un escenario de prueba"""
        print(f"🧪 Evaluando modelo en {test_scenario}...")

        # Debug: mostrar info del preprocesador
        print(
            f"🔍 Preprocesador - Features esperadas: {self.preprocessor.expected_feature_count}"
        )
        print(f"🔍 Preprocesador - Columnas: {self.preprocessor.feature_columns}")

        # Usar RealTimeController en lugar de TrafficLightAgent
        controller = RealTimeController(
            model=self.model,  # Pasar el modelo directamente
            preprocessor=self.preprocessor,  # Pasar el preprocesador directamente
            tls_id=self.tls_id,
            use_gui=False,
        )

        # Asignar modelo y preprocesador directamente
        controller.model = self.model
        controller.preprocessor = self.preprocessor

        collector = DataCollector(self.tls_id)

        metrics = {
            "total_waiting_time": [],
            "average_speed": [],
            "throughput": [],
            "queue_length": [],
            "efficiency_score": [],
        }

        # Iniciar SUMO
        traci.start(["sumo", "-c", test_scenario])

        step = 0
        try:
            while step < evaluation_duration:
                traci.simulationStep()

                # Usar el controller para tomar decisiones
                if step % controller.phase_change_interval == 0:
                    state = controller.get_current_state(step)
                    decisions = controller.make_decision(state)
                    controller.apply_decision(decisions)

                # Recolectar métricas
                current_metrics = self._collect_step_metrics(collector, step)
                for key in metrics.keys():
                    if key in current_metrics:
                        metrics[key].append(current_metrics[key])

                step += 1

        except Exception as e:
            print(f"❌ Error durante evaluación: {e}")
        finally:
            traci.close()

        # Calcular métricas resumen
        summary_metrics = self._calculate_summary_metrics(metrics)
        return summary_metrics

    def _collect_step_metrics(self, collector, step):
        """Recolectar métricas en un step específico - MEJORADO"""
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        try:
            # Usar métodos del DataCollector
            total_waiting = collector.get_total_waiting_time(lanes)
            avg_speed = collector.get_average_speed(lanes)
            throughput = collector.get_throughput(lanes)
            max_queue = collector.get_max_queue_length(lanes)

            # Calcular score de eficiencia con manejo de NaN
            efficiency = self._calculate_efficiency(
                total_waiting, avg_speed, throughput
            )

            return {
                "total_waiting_time": total_waiting,
                "average_speed": avg_speed,
                "throughput": throughput,
                "queue_length": max_queue,
                "efficiency_score": efficiency,
            }
        except Exception as e:
            print(f"⚠️  Error recolectando métricas en step {step}: {e}")
            # Devolver valores por defecto
            return {
                "total_waiting_time": 0,
                "average_speed": 0,
                "throughput": 0,
                "queue_length": 0,
                "efficiency_score": 0,
            }

    def _calculate_efficiency(self, waiting_time, avg_speed, throughput):
        """Calcular score de eficiencia compuesto - MEJORADO"""
        try:
            # Evitar división por cero y NaN
            if waiting_time is None or avg_speed is None or throughput is None:
                return 0.0

            # Normalizar métricas con límites
            norm_waiting = max(0, 1 - (waiting_time / 500)) if waiting_time > 0 else 1.0
            norm_speed = min(1.0, avg_speed / 15.0) if avg_speed > 0 else 0.0
            norm_throughput = min(1.0, throughput / 100.0) if throughput > 0 else 0.0

            # Ponderar métricas
            efficiency = 0.4 * norm_waiting + 0.4 * norm_speed + 0.2 * norm_throughput
            return max(0, min(1, efficiency))
        except:
            return 0.0

    def _calculate_summary_metrics(self, metrics):
        """Calcular métricas resumen de toda la evaluación"""
        summary = {}

        for metric_name, values in metrics.items():
            if values:  # Solo si hay valores
                summary[f"{metric_name}_mean"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)
                summary[f"{metric_name}_max"] = np.max(values)
                summary[f"{metric_name}_min"] = np.min(values)

        # Score final de rendimiento
        summary["final_efficiency_score"] = np.mean(metrics["efficiency_score"])

        return summary

    def compare_with_baseline(self, model_metrics, baseline_metrics):
        """Comparar modelo vs línea base (semáforo tradicional)"""
        comparison = {}

        for metric in model_metrics.keys():
            if metric.endswith("_mean") and metric in baseline_metrics:
                model_val = model_metrics[metric]
                baseline_val = baseline_metrics[metric]

                # Calcular mejora porcentual
                if baseline_val != 0:
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                    comparison[f"{metric}_improvement"] = improvement
                else:
                    comparison[f"{metric}_improvement"] = 0

        return comparison

    def generate_evaluation_report(self, metrics, comparison=None, output_file=None):
        """Generar reporte detallado de evaluación"""
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "tls_id": self.tls_id,
            "metrics": metrics,
            "comparison": comparison or {},
        }

        print("\n" + "=" * 50)
        print("📊 REPORTE DE EVALUACIÓN")
        print("=" * 50)

        print(f"🎯 Score de Eficiencia: {metrics.get('final_efficiency_score', 0):.3f}")
        print(
            f"⏱️  Tiempo de espera promedio: {metrics.get('total_waiting_time_mean', 0):.2f}s"
        )
        print(f"🚗 Velocidad promedio: {metrics.get('average_speed_mean', 0):.2f} m/s")
        print(
            f"📈 Throughput promedio: {metrics.get('throughput_mean', 0):.2f} veh/step"
        )

        if comparison:
            print("\n📈 COMPARACIÓN vs LÍNEA BASE:")
            for metric, improvement in comparison.items():
                if improvement > 0:
                    print(f"  ✅ {metric}: +{improvement:.2f}% de mejora")
                else:
                    print(f"  ❌ {metric}: {improvement:.2f}% de empeoramiento")

        # Guardar reporte si se especifica archivo
        if output_file:
            df_report = pd.DataFrame([report])
            df_report.to_json(output_file, indent=2)
            print(f"\n💾 Reporte guardado en: {output_file}")

        return report


# Función de conveniencia para evaluación rápida
def evaluate_model(model, preprocessor, test_scenario, tls_id="J1"):
    """Función simple para evaluación rápida"""
    evaluator = ModelEvaluator(model, preprocessor, tls_id)
    metrics = evaluator.evaluate_model(test_scenario)
    return metrics
