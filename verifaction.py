# compare_with_metrics.py
import sys

sys.path.append("src")


def run_with_metrics(use_model=True, run_duration=120):
    """Ejecutar simulación mostrando métricas en tiempo real"""
    import traci
    from src.data_collection.data_collectors import DataCollector

    if use_model:
        from src.deployment.traffic_controller import RealTimeController
        from tensorflow.keras.models import load_model  # type: ignore
        from src.data_collection.data_preprocessor import DataPreprocessor

        model_path = "data/trained_models/traffic_model_20251006_1714.h5"
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        preprocessor_path = model_path.replace(".h5", "_preprocessor.pkl")
        preprocessor = DataPreprocessor()
        preprocessor.load(preprocessor_path)

        controller = RealTimeController(
            model=model, preprocessor=preprocessor, tls_id="J8"
        )
        mode = "INTELIGENTE"
    else:
        controller = None
        mode = "TRADICIONAL"

    print(f"\n🚦 Ejecutando semáforo {mode}")
    print("=" * 40)

    traci.start(["sumo-gui", "-c", "prueba2.sumocfg"])
    collector = DataCollector("J8")

    metrics_history = {"waiting_times": [], "speeds": [], "throughputs": []}

    step = 0
    start_time = traci.simulation.getTime()

    while traci.simulation.getTime() - start_time < run_duration:
        traci.simulationStep()

        # Si es modo inteligente, tomar decisión
        if use_model and controller and step % 10 == 0:
            state = controller.get_current_state()
            decision = controller.make_decision(state)
            controller.apply_decision(decision)

        # Recolectar métricas
        data = collector.collect_step_data(step)
        metrics_history["waiting_times"].append(data.get("total_waiting_time", 0))
        metrics_history["speeds"].append(data.get("average_speed", 0))
        metrics_history["throughputs"].append(data.get("throughput", 0))

        # Mostrar progreso cada 50 steps
        if step % 50 == 0:
            avg_waiting = sum(metrics_history["waiting_times"][-50:]) / min(
                50, len(metrics_history["waiting_times"])
            )
            avg_speed = sum(metrics_history["speeds"][-50:]) / min(
                50, len(metrics_history["speeds"])
            )
            avg_throughput = sum(metrics_history["throughputs"][-50:]) / min(
                50, len(metrics_history["throughputs"])
            )

            print(
                f"   Step {step}: Espera={avg_waiting:.1f}s, Velocidad={avg_speed:.2f}m/s, Throughput={avg_throughput:.1f}"
            )

        step += 1

    traci.close()

    # Calcular métricas finales
    final_metrics = {
        "avg_waiting": sum(metrics_history["waiting_times"])
        / len(metrics_history["waiting_times"]),
        "avg_speed": sum(metrics_history["speeds"]) / len(metrics_history["speeds"]),
        "avg_throughput": sum(metrics_history["throughputs"])
        / len(metrics_history["throughputs"]),
    }

    return final_metrics


def main():
    print("📊 COMPARACIÓN CON MÉTRICAS EN TIEMPO REAL")
    print("=" * 50)

    # Ejecutar tradicional
    print("\n1️⃣  Semáforo TRADICIONAL (60 segundos)...")
    traditional_metrics = run_with_metrics(use_model=False, run_duration=240)

    input("\n⏎ Presiona Enter para continuar con semáforo INTELIGENTE...")

    # Ejecutar inteligente
    print("\n2️⃣  Semáforo INTELIGENTE (60 segundos)...")
    smart_metrics = run_with_metrics(use_model=True, run_duration=240)

    # Mostrar comparación
    print("\n" + "=" * 50)
    print("📈 COMPARACIÓN FINAL")
    print("=" * 50)

    print(f"{'Métrica':<15} {'Tradicional':<12} {'Inteligente':<12} {'Mejora':<10}")
    print("-" * 50)

    for metric in ["avg_waiting", "avg_speed", "avg_throughput"]:
        trad = traditional_metrics[metric]
        smart = smart_metrics[metric]

        if metric == "avg_waiting":
            mejora = f"{(trad - smart) / trad * 100:.1f}%" if trad > 0 else "N/A"
            print(f"{'Espera (s)':<15} {trad:<12.1f} {smart:<12.1f} {mejora:<10}")
        elif metric == "avg_speed":
            mejora = f"{(smart - trad) / trad * 100:.1f}%" if trad > 0 else "N/A"
            print(f"{'Velocidad (m/s)':<15} {trad:<12.2f} {smart:<12.2f} {mejora:<10}")
        else:
            mejora = f"{(smart - trad) / trad * 100:.1f}%" if trad > 0 else "N/A"
            print(f"{'Throughput':<15} {trad:<12.1f} {smart:<12.1f} {mejora:<10}")


if __name__ == "__main__":
    main()
