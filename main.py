import traci
import sumolib

def main():
    # Puedes usar sumo-gui para ver la simulación
    traci.start(["sumo-gui", "-c", "prueba2.sumocfg"])
    
    try:
        step = 0
        while step < 1000:
            traci.simulationStep()
            
            
            
            step += 1
    finally:
        traci.close()

def controlar_semaforo_inteligente():
    
    pass

if __name__ == "__main__":
    main()