######################## 
# Calculates the NMSE for three scenarios relative to a ground truth
######################## 

import pymeshlab.pmeshlab
from pymeshlab.pmeshlab import MeshSet
from utils.utils import get_center_of_vertice_pos
import os

# Avoid unnecessary TF logs and configures the gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_num = "" # "" = CPU, 1 = GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

import numpy as np
import time
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RadioMaterial
from sionna.channel import cir_to_ofdm_channel
from sionna.rt import Paths
from sionna.rt import Scene

# Tensorflow seed for reproducibility
tf.random.set_seed(1)

def NMSE(correct_channel: np.ndarray, estimated_channel: np.ndarray):
    # Normalize mean square error (H matrix)
    NMSE = (np.linalg.norm(correct_channel - estimated_channel, ord=2) ** 2) / (
        np.linalg.norm(correct_channel, ord=2) ** 2
    )
    NMSE_dB = 10 * np.log10(NMSE)
    return NMSE_dB

def calculate_nmse_betw_scenarios(ground_truth: list, scenario: list) -> list:
    NMSEs = []
    for i in range(len(scenario)):
        nmse_value = NMSE(ground_truth[i], scenario[i])
        NMSEs.append(nmse_value)
    return NMSEs

def configure_ray_tracing_parameters(scene: Scene) -> None:
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(
        num_rows=int(np.sqrt(nTx)),
        num_cols=int(np.sqrt(nTx)),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_rows=int(np.sqrt(nRx)),
        num_cols=int(np.sqrt(nRx)),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # Defining the custom materials 
    scene.add(custom_material_asphalt)

    # Adjusting some parameters
    scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterial
    scene.synthetic_array = True

def make_ray_tracing(scene: Scene, tx_position_inicial: list = None, rx_position_inicial: list = None) -> Paths:
    if tx_position_inicial is None and rx_position_inicial is None: # 1 Case
        len_path = 0
        tx_position = []
        rx_position = []
        while(len_path < 10 or len_path > 1000):
            # Tx_ Lista de coordenadas
            tx_x = []
            tx_y = []
            tx_z = []

            # Higher Rx and Tx Z
            higher_rx_z = 5 
            higher_tx_z = 0

            # Rx posições (x,y)
            random_rx_x = np.random.uniform(grid_x_start_stop[0], grid_x_start_stop[1])
            random_rx_y = np.random.uniform(grid_y_start_stop[0], grid_y_start_stop[1])

            # Create a meshset
            ms = pymeshlab.pmeshlab.MeshSet()

            # Scenario objects
            for m in os.listdir(original_meshes_path):
                # Load the mesh
                ms.load_new_mesh(os.path.join(original_meshes_path, m))
                cm = ms.current_mesh()
                # Position of the object
                vertice_pos = get_center_of_vertice_pos(cm.vertex_matrix())
                if vertice_pos[2] >= higher_tx_z:
                    tx_x.append(vertice_pos[0])
                    tx_y.append(vertice_pos[1])
                    tx_z.append(vertice_pos[2])
                    higher_tx_z = vertice_pos[2]

                # Rx Z
                if (vertice_pos[0] >= random_rx_x - rx_threshold and vertice_pos[0] <= random_rx_x + rx_threshold) and (vertice_pos[1] >= random_rx_y - rx_threshold and vertice_pos[1] <= random_rx_y + rx_threshold):
                    if vertice_pos[2] > higher_rx_z:
                        higher_rx_z = vertice_pos[2]

            # Max tx_z position
            max_tx_z = max(tx_z) - 2

            # Filtrando os valores de tx_x e tx_y onde tz é o seu valor máximo
            tx_x = [x for x, z in zip(tx_x, tx_z) if z >= max_tx_z]
            tx_y = [y for y, z in zip(tx_y, tx_z) if z >= max_tx_z]

            # Tx (x,y) position where tx_z is MAX
            tx_x = np.random.choice(tx_x)
            tx_y = np.random.choice(tx_y)

            # Tx and Rx Positions
            tx_position = np.array([tx_x, tx_y, higher_tx_z + 5])
            rx_position = np.array([random_rx_x, random_rx_y, higher_rx_z + 1.5])

            # Tx and Rx
            rx = Receiver(name="rx", position=[rx_position[0], rx_position[1], rx_position[2]], orientation=[rx_orientation[0], rx_orientation[1], rx_orientation[2]])
            tx = Transmitter(name="tx", position=[tx_position[0], tx_position[1], tx_position[2]], orientation=[tx_orientation[0], tx_orientation[1], tx_orientation[2]])
            
            # Cam - Bird view
            cam = Camera("cam", position=[random_rx_x - 50, random_rx_y - 50, higher_tx_z + 100], look_at=[0,0,0])
            
            # Avoiding errors
            scene.remove("rx")
            scene.remove("tx")
            scene.remove("cam")

            # Add Tx and Rx
            scene.add(rx)
            scene.add(tx)
            scene.add(cam)
            
            # Cam orientation
            cam.look_at("rx")
            # Calculate the ray-tracing duration
            starting_instant = time.time()
            # Compute propagation paths
            paths = scene.compute_paths(
                max_depth=4, 
                num_samples=1e6,  # Number of rays shot into random directions
                scattering=True,
                diffraction=True,
                reflection=True, 
            )
            ending_instant = time.time()
            print(f"RT duration: {ending_instant-starting_instant}")
            print("Path types: ", paths.types.numpy())
            print("Path len: ", len(paths.types.numpy()[0]))
            len_path = len(paths.types.numpy()[0])

        # Store positions
        tx_x_position.append(tx_position[0])
        tx_y_position.append(tx_position[1])
        tx_z_position.append(tx_position[2])
        rx_x_position.append(rx_position[0])
        rx_y_position.append(rx_position[1])
        rx_z_position.append(rx_position[2])

        # Distance RX TX
        distance = np.sqrt((tx_position[0] - rx_position[0])**2 + (tx_position[1] - rx_position[1])**2 + (tx_position[2] - rx_position[2])**2)
        tx_rx_distance.append(distance)
        
        # IMG
        scene.render_to_file(camera="cam", filename="mitsubas/nmse/scenario_img" + str(i) + str(j) + str(k) + ".png", paths=paths, resolution=[500, 500])
    else: # 2 case
        # Posições iniciais
        tx_position = tx_position_inicial
        rx_position = rx_position_inicial

        # Tx and Rx
        rx = Receiver(name="rx", position=[rx_position[0], rx_position[1], rx_position[2]], orientation=[rx_orientation[0], rx_orientation[1], rx_orientation[2]])
        tx = Transmitter(name="tx", position=[tx_position[0], tx_position[1], tx_position[2]], orientation=[tx_orientation[0], tx_orientation[1], tx_orientation[2]])
        
        # Cam - Bird view
        cam = Camera("cam", position=[rx_position[0] - 50, rx_position[1] - 50, tx_position[2] + 100], look_at=[0,0,0])
        
        # Avoiding errors
        scene.remove("rx")
        scene.remove("tx")
        scene.remove("cam")

        # Add Tx and Rx
        scene.add(rx)
        scene.add(tx)
        scene.add(cam)

        # Cam orientation
        cam.look_at("rx")
        # Calculate the ray-tracing duration
        starting_instant = time.time()
        # Compute propagation paths
        paths = scene.compute_paths(
            max_depth=4,
            num_samples=1e6,  # Number of rays shot into random directions
            scattering=True,
            diffraction=True,
            reflection=True, 
        )
        ending_instant = time.time()
        print(f"RT duration (simplified): {ending_instant-starting_instant}")
        print("Path types: ", paths.types.numpy())
        print("Path len: ", len(paths.types.numpy()[0]))
        # IMG
        scene.render_to_file(camera="cam", filename="mitsubas/nmse/simplified_scenario_img" + str(i) + str(j) + str(k) + ".png", paths=paths, resolution=[500, 500])

    return paths, tx_position, rx_position

def calculate_h_freq(paths: Paths):
    # Returns the channel impulse response in the form of path coefficients (a) and path delay (tau)
    a, tau = paths.cir()
    # Only frequency 0
    frequencies = np.array([0], dtype=np.float32)
    # Compute the frequency response of the channel at frequencies
    h_freq = cir_to_ofdm_channel(
        frequencies, a, tau, normalize=True
    ) 
    h_freq = np.squeeze(h_freq, axis = (0,1,3,5,6))
    return h_freq

def save_nmse_betw_scenes_npz():
    # NMSE calculation between the scenarios and the ground truth
    NMSE_1 = calculate_nmse_betw_scenarios(hs_freq_ground_truth, hs_freq_scene_1)

    # save files
    np.savez("mitsubas/nmse/nmses.npz", 
             NMSE_1=NMSE_1)

def save_positions_npz():
    np.savez("mitsubas/nmse/nmses_positions.npz", tx_x_position=tx_x_position, tx_y_position=tx_y_position, tx_z_position=tx_z_position, 
    rx_x_position=rx_x_position, rx_y_position=rx_y_position, rx_z_position=rx_z_position, tx_rx_distance=tx_rx_distance)

if __name__ == "__main__":
    # Path to mitsubas
    cwd = os.getcwd()

    # Modern City coordinates
    lim_sup_esq = (-105.4, 91)
    lim_sup_dir = (78.5, 91)
    lim_inf_esq = (-105.4, -106)
    lim_inf_dir = (78.5, -106)

    # Positions and distance
    tx_rx_distance = []
    tx_x_position = []
    tx_y_position = []
    tx_z_position = []
    rx_x_position = []
    rx_y_position = []
    rx_z_position = []
 
    # Parameters to calculate the NMSE between scenes and the ground truth
    hs_freq_ground_truth = []
    hs_freq_scene_1 = []

    # Scene parameters (Orientations)
    tx_orientation = [6, 0.5, 0]
    rx_orientation = [0, 0, 0]

    # Parameter to define the number of txs and rxs in the antenna array
    nTx = 1      
    nRx = 1

    # Loop
    rx_threshold = 2.5
    i_range = 10 
    for i in range(i_range):
        # grid x and y, start, stop and range > Number of loops
        gd = 50 + (i*10)
        
        # Grids size
        j_range = np.ceil((np.abs(lim_sup_esq[0]) + np.abs(lim_sup_dir[0]))/gd).astype(int)
        k_range = np.ceil((np.abs(lim_sup_esq[1]) + np.abs(lim_inf_dir[1]))/gd).astype(int)

        # Debug
        print(f"Simulations com i: {i_range}, grid: {j_range} x {k_range}, Total simulations: {j_range * k_range}")

        for j in range(j_range):
            # Grid config (X) (start_stop)
            grid_x_start_stop = ((j*gd)+ lim_sup_esq[0], ((j+1)*gd) + lim_sup_esq[0])
            for k in range(k_range):
                print(f"Simulation: ", (j*10 + k*1), "\n")
                print(f"Scenario {i}{j}{k}")
                # Grid config (Y) (start_stop)
                grid_y_start_stop = (lim_sup_esq[1] - (k*gd), lim_sup_esq[1] - ((k+1)*gd))

                # Select the mitsuba pairs
                original_mitsuba = cwd +   "/mitsubas/scenarios/scenario" + str(i) + str(j) + str(k) + "/modern_export.xml"
                simplified_mitsuba = cwd + "/mitsubas/simplified_scenarios_3/scenario" + str(i) + str(j) + str(k) + "/modern_export.xml"
                
                # Original Meshes Path
                original_meshes_path = cwd + "/mitsubas/scenarios/scenario" + str(i) + str(j) + str(k) + "/meshes"
                
                # Debug
                print(f"Original path: {original_mitsuba} \nSimplified Path: {simplified_mitsuba}")

                # Defining the custom materials
                custom_material_asphalt = RadioMaterial("asphalt" ,5.72 ,5e-4)  

                # Ground truth ----------------------------------------------------------
                print("\nLoading ground truth scene ...")
                original_scene = load_scene(original_mitsuba)
                configure_ray_tracing_parameters(original_scene)
                print("Original Scene: ")   
                
                # Simulation_01
                # Compute the ray tracing paths
                paths, tx_position_1, rx_position_1 = make_ray_tracing(original_scene)
                # Find the channel h_freq
                h_freq = calculate_h_freq(paths)
                # Store it to NMSE calculation
                hs_freq_ground_truth.append(h_freq)

                # Simulation_02
                # Compute the ray tracing paths
                paths, tx_position_2, rx_position_2 = make_ray_tracing(original_scene)
                # Find the channel h_freq
                h_freq = calculate_h_freq(paths)
                # Store it to NMSE calculation
                hs_freq_ground_truth.append(h_freq)

                # simplification_1 part --------------------------------------------------
                print("\nLoading scene 1 ...")
                simplified_scene = load_scene(simplified_mitsuba)
                configure_ray_tracing_parameters(simplified_scene)
                print("Scene 1: ")

                # Simulation_01
                # Compute the ray tracing paths
                paths, _, _ = make_ray_tracing(simplified_scene, tx_position_1, rx_position_1)
                # Find the channel h_freq
                h_freq = calculate_h_freq(paths)
                # Store it to NMSE calculation
                hs_freq_scene_1.append(h_freq)

                 # Simulation_02
                # Compute the ray tracing paths
                paths, _, _ = make_ray_tracing(simplified_scene, tx_position_2, rx_position_2)
                # Find the channel h_freq
                h_freq = calculate_h_freq(paths)
                # Store it to NMSE calculation
                hs_freq_scene_1.append(h_freq)

                print("\nSalvando os NMSEs e positions do cenário " + str(i) + str(j) + str(k) + "\n")
                # Save all in an NPZ
                save_nmse_betw_scenes_npz()
                save_positions_npz()