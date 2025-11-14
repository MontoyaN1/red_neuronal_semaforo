# unified_evaluator.py
import traci
import numpy as np
import sys
import os
from datetime import datetime
import json

# Añadir path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_collection.data_collectors import DataCollector
from src.deployment.traffic_controller import RealTimeController
from src.data_collection.data_preprocessor import DataPreprocessor
from tensorflow.keras.models import load_model  # type: ignore


class ModelEvaluator:
    def __init__(self, tls_id="J1"):
        self.tls_id = tls_id
        self.model = None
        self.preprocessor = None
        self.controller = None

    def load_model_from_file(self, model_path):
        """Cargar modelo y preprocesador desde archivos .h5 y .pkl"""
        print(f"📦 Cargando modelo desde: {model_path}")

        # Cargar modelo
        self.model = load_model(model_path, compile=False)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Cargar preprocesador
        preprocessor_path = model_path.replace(".h5", "_preprocessor.pkl")
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load(preprocessor_path)

        # Crear controller
        self.controller = RealTimeController(
            model=self.model, preprocessor=self.preprocessor, tls_id=self.tls_id
        )

        print(f"✅ Modelo cargado: {self.preprocessor.expected_feature_count} features")

    def set_model_directly(self, model, preprocessor):
        """Configurar modelo directamente (para uso en pipeline)"""
        self.model = model
        self.preprocessor = preprocessor
        self.controller = RealTimeController(
            model=model, preprocessor=preprocessor, tls_id=self.tls_id
        )

    def run_comparison(self, test_scenario, run_duration=240, use_gui=False):
        """Ejecutar comparación A/B entre modelo inteligente y tradicional"""
        print("📊 EJECUTANDO COMPARACIÓN A/B")
        print("=" * 50)

        # 1. Ejecutar semáforo tradicional
        print("\n1️⃣  Semáforo TRADICIONAL...")
        traditional_metrics = self._run_simulation(
            use_model=False,
            test_scenario=test_scenario,
            run_duration=run_duration,
            use_gui=use_gui,
            mode="TRADICIONAL",
        )

        input("\n⏎ Presiona Enter para continuar con semáforo INTELIGENTE...")

        # 2. Ejecutar semáforo inteligente
        print("\n2️⃣  Semáforo INTELIGENTE...")
        intelligent_metrics = self._run_simulation(
            use_model=True,
            test_scenario=test_scenario,
            run_duration=run_duration,
            use_gui=use_gui,
            mode="INTELIGENTE",
        )

        # 3. Generar reporte comparativo
        comparison_report = self._generate_comparison_report(
            traditional_metrics, intelligent_metrics
        )

        return comparison_report

    def _run_simulation(self, use_model, test_scenario, run_duration, use_gui, mode):
        """Ejecutar una simulación individual"""
        print(f"🚦 Ejecutando semáforo {mode}")
        print("-" * 40)

        # Configurar SUMO (con o sin GUI)
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        traci.start([sumo_binary, "-c", test_scenario])

        collector = DataCollector(self.tls_id)
        metrics_history = {
            "waiting_times": [],
            "speeds": [],
            "throughputs": [],
            "queue_lengths": [],
            "efficiency_scores": [],
        }

        step = 0
        start_time = traci.simulation.getTime()

        try:
            while traci.simulation.getTime() - start_time < run_duration:
                traci.simulationStep()

                # Tomar decisión si es modo inteligente
                if use_model and self.controller and step % 10 == 0:
                    state = self.controller.get_current_state()
                    decision = self.controller.make_decision(state)
                    self.controller.apply_decision(decision)

                # Recolectar métricas
                data = collector.collect_step_data(step)
                current_metrics = self._extract_metrics_from_data(data)

                for key in metrics_history.keys():
                    if key in current_metrics:
                        metrics_history[key].append(current_metrics[key])

                # Mostrar progreso en tiempo real
                if step % 50 == 0:
                    self._print_real_time_metrics(metrics_history, step, mode)

                step += 1

        except Exception as e:
            print(f"❌ Error en simulación {mode}: {e}")
        finally:
            traci.close()

        # Calcular métricas finales
        final_metrics = self._calculate_final_metrics(metrics_history)
        return final_metrics

    def _extract_metrics_from_data(self, data):
        """Extraer y calcular métricas desde los datos del collector"""
        waiting_time = data.get("total_waiting_time", 0)
        avg_speed = data.get("average_speed", 0)
        throughput = data.get("throughput", 0)
        max_queue = data.get("max_queue_length", 0)

        # Calcular score de eficiencia simple
        efficiency = self._calculate_efficiency_score(
            waiting_time, avg_speed, throughput
        )

        return {
            "waiting_times": waiting_time,
            "speeds": avg_speed,
            "throughputs": throughput,
            "queue_lengths": max_queue,
            "efficiency_scores": efficiency,
        }

    def _calculate_efficiency_score(self, waiting, speed, throughput):
        """Calcular score de eficiencia simplificado"""
        if waiting == 0 or speed == 0:
            return 0.0

        # Fórmula simple: priorizar reducción de tiempo de espera
        norm_waiting = 1 / (1 + waiting / 100)  # Normalizar waiting time
        norm_speed = speed / 15.0  # Asumiendo 15 m/s como velocidad máxima típica
        norm_throughput = throughput / 50.0  # Asumiendo 50 vehículos como máximo típico

        # Ponderación: 60% waiting, 30% speed, 10% throughput
        efficiency = 0.6 * norm_waiting + 0.3 * norm_speed + 0.1 * norm_throughput
        return max(0, min(1, efficiency))

    def _print_real_time_metrics(self, metrics_history, step, mode):
        """Mostrar métricas en tiempo real"""
        # Calcular promedios de los últimos 50 steps
        avg_waiting = (
            np.mean(metrics_history["waiting_times"][-50:])
            if metrics_history["waiting_times"]
            else 0
        )
        avg_speed = (
            np.mean(metrics_history["speeds"][-50:]) if metrics_history["speeds"] else 0
        )
        avg_throughput = (
            np.mean(metrics_history["throughputs"][-50:])
            if metrics_history["throughputs"]
            else 0
        )
        avg_efficiency = (
            np.mean(metrics_history["efficiency_scores"][-50:])
            if metrics_history["efficiency_scores"]
            else 0
        )

        print(
            f"   {mode[:3]} Step {step}: "
            f"Espera={avg_waiting:.1f}s | "
            f"Velocidad={avg_speed:.2f}m/s | "
            f"Throughput={avg_throughput:.1f} | "
            f"Eficiencia={avg_efficiency:.3f}"
        )

    def _calculate_final_metrics(self, metrics_history):
        """Calcular métricas finales de la simulación"""
        return {
            "avg_waiting": np.mean(metrics_history["waiting_times"])
            if metrics_history["waiting_times"]
            else 0,
            "avg_speed": np.mean(metrics_history["speeds"])
            if metrics_history["speeds"]
            else 0,
            "avg_throughput": np.mean(metrics_history["throughputs"])
            if metrics_history["throughputs"]
            else 0,
            "avg_queue_length": np.mean(metrics_history["queue_lengths"])
            if metrics_history["queue_lengths"]
            else 0,
            "avg_efficiency": np.mean(metrics_history["efficiency_scores"])
            if metrics_history["efficiency_scores"]
            else 0,
            "total_steps": len(metrics_history["waiting_times"]),
        }

    def _generate_comparison_report(self, traditional_metrics, intelligent_metrics):
        """Generar reporte comparativo detallado"""
        print("\n" + "=" * 60)
        print("📈 REPORTE COMPARATIVO FINAL")
        print("=" * 60)

        report = {
            "evaluation_date": datetime.now().isoformat(),
            "tls_id": self.tls_id,
            "traditional_metrics": traditional_metrics,
            "intelligent_metrics": intelligent_metrics,
            "improvements": {},
        }

        # Calcular mejoras
        improvements = {}
        for metric in ["avg_waiting", "avg_speed", "avg_throughput", "avg_efficiency"]:
            trad = traditional_metrics[metric]
            smart = intelligent_metrics[metric]

            if metric == "avg_waiting" and trad > 0:
                improvement_pct = ((trad - smart) / trad) * 100
                improvements[metric] = improvement_pct
            elif trad > 0:
                improvement_pct = ((smart - trad) / trad) * 100
                improvements[metric] = improvement_pct
            else:
                improvements[metric] = 0

        report["improvements"] = improvements

        # Mostrar tabla comparativa
        print(f"{'MÉTRICA':<25} {'TRADICIONAL':<12} {'INTELIGENTE':<12} {'MEJORA':<10}")
        print("-" * 60)

        metrics_display = {
            "avg_waiting": ("Tiempo Espera (s)", True),  # True = menor es mejor
            "avg_speed": ("Velocidad (m/s)", False),  # False = mayor es mejor
            "avg_throughput": ("Throughput", False),
            "avg_efficiency": ("Eficiencia", False),
        }

        for metric, (display_name, lower_better) in metrics_display.items():
            trad = traditional_metrics[metric]
            smart = intelligent_metrics[metric]
            improvement = improvements[metric]

            if lower_better:
                trend = "↓" if improvement > 0 else "↑"
            else:
                trend = "↑" if improvement > 0 else "↓"

            print(
                f"{display_name:<25} {trad:<12.2f} {smart:<12.2f} {trend} {improvement:>5.1f}%"
            )

        # Resumen ejecutivo
        print("\n🎯 RESUMEN EJECUTIVO:")
        if improvements["avg_efficiency"] > 0:
            print(
                f"✅ Eficiencia general mejorada: +{improvements['avg_efficiency']:.1f}%"
            )
        if improvements["avg_waiting"] > 0:
            print(f"✅ Tiempo de espera reducido: -{improvements['avg_waiting']:.1f}%")
        if improvements["avg_speed"] > 0:
            print(f"✅ Velocidad aumentada: +{improvements['avg_speed']:.1f}%")

        return report

    def save_report(self, report, output_file=None):
        """Guardar reporte en archivo JSON"""
        if output_file is None:
            output_file = (
                f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"💾 Reporte guardado en: {output_file}")
        return output_file


# FUNCIONES DE CONVENIENCIA PARA DIFERENTES USOS


def evaluate_model_file(model_path, test_scenario, tls_id="J1", use_gui=False):
    """Evaluar un modelo desde archivo .h5 (uso independiente)"""
    evaluator = ModelEvaluator(tls_id=tls_id)
    evaluator.load_model_from_file(model_path)
    report = evaluator.run_comparison(test_scenario, use_gui=use_gui)
    return report


def evaluate_model_direct(model, preprocessor, test_scenario, tls_id="J1"):
    """Evaluar modelo directamente (para uso en pipeline)"""
    evaluator = ModelEvaluator(tls_id=tls_id)
    evaluator.set_model_directly(model, preprocessor)
    report = evaluator.run_comparison(test_scenario)
    return report


# MODO INDEPENDIENTE
if __name__ == "__main__":
    # Ejemplo de uso independiente
    model_path = (
        "data/trained_models/traffic_model_20251006_1714.h5"  # Cambiar por tu modelo
    )
    test_scenario = "prueba2.sumocfg"

    print("🔍 EVALUADOR UNIFICADO - MODO INDEPENDIENTE")
    report = evaluate_model_file(model_path, test_scenario, use_gui=True)

    # Guardar reporte
    evaluator = ModelEvaluator()
    evaluator.save_report(report)
