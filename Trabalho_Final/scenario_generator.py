import pymeshlab.pmeshlab
from pymeshlab.pmeshlab import MeshSet
from tqdm import tqdm
from re import sub
import argparse
import numpy as np
import shutil
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RadioMaterial
from utils.utils import remove_lines_from_xml
import os
import tensorflow as tf

# Tensorflow and gpu parameters
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_num = ""
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

class Simplification:
    def __init__(self, ms: MeshSet) -> None:
        self.ms = ms

    def simplification_algorithm(self) -> MeshSet:
        # Apply the algorithm
        if args.simplification_method == "quadric":
            #print("Using the quadric edge collapse simplification_method ...")
            n_faces = self.ms.current_mesh().face_number()
            if n_faces > number_face_collapse: # Only objects with more than 15 faces (objects with less than 15 faces are destroyed by the decimate)
                self.ms.meshing_decimation_quadric_edge_collapse(targetperc=args.parameter)
        elif args.simplification_method == "vertex":
            #print("Using the vertex clustering simplification_method ...")
            self.ms.meshing_decimation_clustering(threshold=pymeshlab.pmeshlab.PercentageValue(args.parameter))

    def no_cut(self, out: str) -> None:
        self.simplification_algorithm()
        self.ms.save_current_mesh(out)
    
    def scene_generator(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str) -> None:
        vertice_pos = get_center_of_vertice_pos(vertice_pos)
        # Mechanisms to divide the scenario into grids 
        if (vertice_pos[0] >= lim_sup_esq[0] + grid_x_start_stop[0] and vertice_pos[0] <= lim_sup_esq[0] + grid_x_start_stop[1]) and (vertice_pos[1] >= lim_sup_esq[1] + grid_y_start_stop[0] and vertice_pos[1] <= lim_sup_esq[1] + grid_y_start_stop[1]):
            self.simplification_algorithm()
            self.ms.save_current_mesh(out)
            return None
        else:
            # Cut 
            remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def expansion_cut(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str) -> None:
        # Expansion Method
        if ((np.abs(vertice_pos[0][0] - tx[0]) < c) and (np.abs(vertice_pos[0][1] - tx[1]) < c) and (np.abs(vertice_pos[0][2] - tx[2]) < c)) or ((np.abs(vertice_pos[0][0] - rx[0]) < c) and (np.abs(vertice_pos[0][1] - rx[1]) < c) and (np.abs(vertice_pos[0][2] - rx[2]) < c)):
            self.simplification_algorithm()
            self.ms.save_current_mesh(out)
        else:
            # Cut 
            remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def square_cut(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str) -> None:
        if(tx[0] > rx[0]): # Tx[0] > Rx[0]
            if(tx[1] > rx[1]): # Tx[1] > Rx[1]
                if((vertice_pos[0][0] < tx[0]+c and vertice_pos[0][0] > rx[0]-c) and (vertice_pos[0][1] < tx[1]+c and vertice_pos[0][1] > rx[1]-c)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
            else: # Tx[1] < Rx[1]
                if((vertice_pos[0][0] < tx[0]+c and vertice_pos[0][0] > rx[0]-c) and (vertice_pos[0][1] > tx[1]-c and vertice_pos[0][1] < rx[1]+c)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
        else: # Tx[0] < Rx[0]
            if(tx[1] > rx[1]): # Tx[1] > Rx[1]
                if((vertice_pos[0][0] > tx[0]-c and vertice_pos[0][0] < rx[0]+c) and (vertice_pos[0][1] < tx[1]+c and vertice_pos[0][1] > rx[1]-c)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
            else: # Tx[1] < Rx[1]
                if((vertice_pos[0][0] > tx[0]-c and vertice_pos[0][0] < rx[0]+c) and (vertice_pos[0][1] > tx[1]-c and vertice_pos[0][1] < rx[1]+c)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def square_cut_distance(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str, object_name: str, position_dict_multi_material: dict) -> None:
        vertice_pos = adjust_positions_in_multi_material_scenario(vertice_pos, object_name, position_dict_multi_material)
        # Distance
        d = np.sqrt((tx[0] - rx[0])**2 + (tx[1] - rx[1])**2 + (tx[2] - rx[2])**2) # always calculates ?
        if d < 15: # Anything Less than this almost destroys completely the scene
            d += 10
        d = d/2
        print(f'distance: ', d)
        if(tx[0] > rx[0]): # Tx[0] > Rx[0]
            if(tx[1] > rx[1]): # Tx[1] > Rx[1]
                if((vertice_pos[0] < tx[0]+d and vertice_pos[0] > rx[0]-d) and (vertice_pos[1] < tx[1]+d and vertice_pos[1] > rx[1]-d)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
            else: # Tx[1] < Rx[1]
                if((vertice_pos[0] < tx[0]+d and vertice_pos[0] > rx[0]-d) and (vertice_pos[1] > tx[1]-d and vertice_pos[1] < rx[1]+d)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
        else: # Tx[0] < Rx[0]
            if(tx[1] > rx[1]): # Tx[1] > Rx[1]
                if((vertice_pos[0] > tx[0]-d and vertice_pos[0] < rx[0]+d) and (vertice_pos[1] < tx[1]+d and vertice_pos[1] > rx[1]-d)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
            else: # Tx[1] < Rx[1]
                if((vertice_pos[0] > tx[0]-d and vertice_pos[0] < rx[0]+d) and (vertice_pos[1] > tx[1]-d and vertice_pos[1] < rx[1]+d)):
                    self.simplification_algorithm()
                    self.ms.save_current_mesh(out)
                else:
                    # Cut 
                    remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def sphere_cut(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str, object_name: str, position_dict_multi_material: dict) -> None:
        vertice_pos = adjust_positions_in_multi_material_scenario(vertice_pos, object_name, position_dict_multi_material)
        # Distance between tx and rx  
        d_tx_rx = np.sqrt((tx[0] - rx[0])**2 + (tx[1] - rx[1])**2 + (tx[2] - rx[2])**2) # always calculates ?
        # Sphere's center
        center = np.array([(tx[0] + rx[0])/2, (tx[1] + rx[1])/2, (tx[2] + rx[2])/2]) 
        # Distance between the current 3D object and the center 
        d = np.sqrt((vertice_pos[0] - center[0])**2 + (vertice_pos[1] - center[1])**2 + (vertice_pos[2] - center[2])**2) 

        if d < d_tx_rx: # Within the sphere's radius
            self.simplification_algorithm()
            self.ms.save_current_mesh(out)
        else:
            # Cut 
            remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def coveragemap_cut(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str, cm_data_list: list[tuple[np.float32, np.ndarray]], cell_size: float, object_name: str, position_dict_multi_material: dict) -> None:
        dB_value = -140 # Threshold in dB to cut
        vertice_pos = adjust_positions_in_multi_material_scenario(vertice_pos, object_name, position_dict_multi_material)
        for power, coords in cm_data_list: # iterates through all cmap values
            if vertice_pos[0] > coords[0] - cell_size and vertice_pos[0] < coords[0] + cell_size and vertice_pos[1] > coords[1] - cell_size  and vertice_pos[1] < coords[1] + cell_size and power > dB_value:
                self.simplification_algorithm() # Save and simply
                self.ms.save_current_mesh(out)
                return None
        
        # Check if the object is the ground
        if object_name == "ground.ply" or object_name == "mesh-Plane.ply":
            self.simplification_algorithm() # Save if yes
            self.ms.save_current_mesh(out)
        else:
            # Cut 
            remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

    def interactions_cut(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str, object_name: str, position_dict_multi_material: dict, interactions: list) -> None:
        vertice_pos = adjust_positions_in_multi_material_scenario(vertice_pos, object_name, position_dict_multi_material)
        threshold = 2
        for coords in interactions:
            if vertice_pos[0] > coords[0] - threshold and vertice_pos[0] < coords[0] + threshold and vertice_pos[1] > coords[1] - threshold  and vertice_pos[1] < coords[1] + threshold and vertice_pos[2] > coords[2] - threshold  and vertice_pos[2] < coords[2] + threshold:
                self.simplification_algorithm() # Save and simply
                self.ms.save_current_mesh(out)
                return None
        # Cut 
        remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)
                

def adjust_positions_in_multi_material_scenario(vertice_pos: np.ndarray, object_name: str, position_dict_multi_material: dict) -> np.ndarray:
    new_center_of_vertice_pos = None
    # Remove the suffix to compare with dict keys
    pattern =  r'-itu_.*'
    new_object_name = sub(pattern, "", object_name)

    for chave in position_dict_multi_material:
        if new_object_name in chave:
            # If the object already is mapped, we use this value, avoiding using two different positions for a house and its roof
            new_center_of_vertice_pos = position_dict_multi_material[chave]
            return new_center_of_vertice_pos
    # If object is not mapped, we get the center position and store it in the dict
    new_center_of_vertice_pos = get_center_of_vertice_pos(vertice_pos)
    position_dict_multi_material[object_name] = new_center_of_vertice_pos
    return new_center_of_vertice_pos

def get_center_of_vertice_pos(vertice_pos: np.ndarray) -> np.ndarray:
    vertice_matrix_len = len(vertice_pos)
    # The first position in the vertex matrix
    first_position = vertice_pos[0]
    highest_pos = vertice_pos[0]
    highest_distance = 0 

    for i in range(vertice_matrix_len - 1):
        # Distance between the first vertex in the 3D object and the next 
        distance = np.sqrt((vertice_pos[0][0] - vertice_pos[i+1][0])**2 + (vertice_pos[0][1] - vertice_pos[i+1][1])**2 + (vertice_pos[0][2] - vertice_pos[i+1][2])**2) 
        if distance > highest_distance:
            # If greater, store the values
            highest_distance = distance
            highest_pos = vertice_pos[i+1]

    vertice_pos_center = np.array([(first_position[0] + highest_pos[0])/2, (first_position[1] + highest_pos[1])/2, (first_position[2] + highest_pos[2])/2])
    return vertice_pos_center 

def parse_arguments() -> argparse.Namespace:
    # Choose the simplification type
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplification_method","-m", required=False, type=str, help="Choose the algorithm to simplify (vertex, quadric)")
    parser.add_argument("--parameter","-p", required=False, type=float, help="Choose the algorithm paramater to simplify (vertex: Cell Size, the size of the cell of the clustering grid; quadric: Percentage reduction (0..1), if non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size.)")
    parser.add_argument("--thresholdCut","-c", required=False, default=50 ,type=float, help="Threshold to cut if the simplification type is expansion or square_with_c, default=50")
    parser.add_argument("--cut_type","-ct", required=True, type=str, help="Choose the type of cut (square, square_with_distance, expansion, sphere, cmap, interactions, no_cut, scene generator)")
    return parser.parse_args()

def compute_paths_interactions_function():
    # Scene parameters (Orientations)
    tx_orientation = [6, 0.5, 0]
    rx_orientation = [0, 0, 0]

    # Antenna arrays parameters 
    nTx = 64
    nRx = 4

    # Load original scene (without cut)
    print("Starting scene loading ...")
    scene = load_scene(original_xml_path)

    # Defining the custom materials
    custom_material_asphalt = RadioMaterial("asphalt", 5.72, 5e-4)
    scene.add(custom_material_asphalt)

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
    # Create transmitter
    tx_1 = Transmitter(
        name="tx", position=[tx[0], tx[1], tx[2]], orientation=[tx_orientation[0], tx_orientation[1], tx_orientation[2]]
    )
    # Add transmitter instance to scene
    scene.add(tx_1)

    # Create a receive
    rx_1 = Receiver(
        name="rx",
        position=[rx[0], rx[1], rx[2]],
        orientation=[rx_orientation[0], rx_orientation[0], rx_orientation[0]],
    )
    # Add receiver instance to scene
    scene.add(rx_1)

    scene.frequency = 40e9  # in Hz; implicitly updates RadioMaterial
    scene.synthetic_array = True 

    paths = scene.compute_paths(max_depth=5, num_samples=int(1e6), 
                                diffraction=True, scattering=True,
                                reflection=True)

    interactions = []
    for value in tf.reshape(paths.vertices, [-1,3]):
        interactions.append(value.numpy())
    
    return interactions
    
def coverage_map_function() -> tuple[list[tuple[np.float32, np.ndarray]], float]:
    # List with cm power and position values
    cmap_data_list = []

    # Scene parameters (Orientations)
    tx_orientation = [6, 0.5, 0]
    rx_orientation = [0, 0, 0]

    # Cell Size (cmap)
    cell_size = 10.

    # Antenna arrays parameters 
    nTx = 64
    nRx = 4

    # Load original scene (without cut)
    print("Starting scene loading ...")
    scene = load_scene(original_xml_path)

    # Defining the custom materials
    custom_material_asphalt = RadioMaterial("asphalt", 5.72, 5e-4)
    scene.add(custom_material_asphalt)

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
    # Create transmitter
    tx_1 = Transmitter(
        name="tx", position=[tx[0], tx[1], tx[2]], orientation=[tx_orientation[0], tx_orientation[1], tx_orientation[2]]
    )
    # Add transmitter instance to scene
    scene.add(tx_1)

    # Create a receive
    rx_1 = Receiver(
        name="rx",
        position=[rx[0], rx[1], rx[2]],
        orientation=[rx_orientation[0], rx_orientation[0], rx_orientation[0]],
    )
    # Add receiver instance to scene
    scene.add(rx_1)

    scene.frequency = 40e9  # in Hz; implicitly updates RadioMaterial
    scene.synthetic_array = True

    print("Starting coverage map ...")
    cmap = scene.coverage_map(max_depth=5,
                    #diffraction=True, # Disable to see the effects of diffraction
                    cm_cell_size=(cell_size, cell_size), # ) # Grid size of coverage map cells in m
                    #combining_vec=None,
                    #precoding_vec=None,
                    num_samples=int(1e6)) # Reduce if your hardware does not have enough memory
    
    power_levels = np.log10(cmap.as_tensor()) * 10 # Power in dB
    global_coordinates = cmap.cell_centers.numpy() # Coordinates in blender

    for x in range(cmap.num_cells_y): # Number of y_cells (cmap)
        for y in range(cmap.num_cells_x): # Number of x_cells (cmap)
            power_level = power_levels[0, x, y] # 0 = 1 tx
            coordinates = global_coordinates[x, y, :]
            cmap_data_list.append((power_level, coordinates))
    
    return cmap_data_list, cell_size

def simplification_process(ms: MeshSet) -> None:
    # Adjust the output folder (xml + clean meshes)
    if not os.path.exists(new_folder_path):
    # Create scenario folder, if it doesn't exist
        os.makeris(new_folder_path)
    # Copy the original XML
    shutil.copy(original_xml_path, new_xml_path)
    if not os.path.exists(new_meshes_path):
    # Create meshes folder, if it doesn't exist
        os.makedirs(new_meshes_path)
    for _, _, files in os.walk(os.path.join(new_folder_path, "meshes")):
        for file in files: 
            os.remove(os.path.join(new_meshes_path, file))
            
    print("Starting the simplification ...")
    # Create a simplification class
    simplification = Simplification(ms)
    # Avoiding errors (ignore)
    cmap_data_list, cell_size = None, None
    # Debug and choose of the simplification type
    if args.cut_type == "square":
        print("Using the square with threshold type")
        print("c = ", c)
        simplify_and_cut_function = simplification.square_cut
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml"]
    elif args.cut_type == "square_with_distance":
        print("Using the square with distance type")
        simplify_and_cut_function = simplification.square_cut_distance
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml", "object_name", "position_dict_multi_material"]
    elif args.cut_type == "expansion":
        print("Using the expansion type")
        print("c = ", c)
        simplify_and_cut_function = simplification.expansion_cut
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml"]
    elif args.cut_type == "sphere":
        print("Using the sphere type")
        simplify_and_cut_function = simplification.sphere_cut
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml", "object_name", "position_dict_multi_material"]
    elif args.cut_type == "cmap":
        print("Using coverage map type")
        # Make the coverage map
        cmap_data_list, cell_size = coverage_map_function()
        simplify_and_cut_function = simplification.coveragemap_cut
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml", "cmap_data_list", "cell_size", "object_name", "position_dict_multi_material"]
    elif args.cut_type == "interactions":
        simplify_and_cut_function = simplification.interactions_cut
        # Find the interactions
        interactions = compute_paths_interactions_function()
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml", "object_name", "position_dict_multi_material", "interactions"]
    elif args.cut_type == "no_cut":
        print("Without cut type")
        simplify_and_cut_function = simplification.no_cut
        parameters = ["out"]
    elif args.cut_type == "scene_generator":
        print("Running the Scene Generator")
        simplify_and_cut_function = simplification.scene_generator
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml"]
    else:
        raise Exception("Simplification type invalid")

    # Dict to correct the position of objects that should be one, but are multiple
    position_dict_multi_material = {}

    for m in tqdm(os.listdir(original_meshes_path), colour="green"):
        # Load the mesh
        ms.load_new_mesh(os.path.join(original_meshes_path, m))
        cm = ms.current_mesh()
        # Position of the object
        vertice_pos = cm.vertex_matrix()
        # Save the mesh
        out = os.path.join("mitsubas/new_test/meshes", str(m))
        # To avoid cut the floor
        object_name = str(m)
        # String to identify objects in xml to cut
        keyword_to_cut_xml = "/" + str(m)

        # Associate strings with the real parameters
        parameters_dict = {
            "out": out,
            "vertice_pos": vertice_pos,
            "keyword_to_cut_xml": keyword_to_cut_xml,
            "cmap_data_list": cmap_data_list,
            "cell_size": cell_size,
            "object_name": object_name,
            "position_dict_multi_material": position_dict_multi_material,
            "interactions": interactions
        }
        arg = [parameters_dict[arg] for arg in parameters]

        # Simplify and cut objects
        simplify_and_cut_function(*arg)

if __name__ == "__main__":
    # Argparses
    args = parse_arguments()

    # Threshold to cut with specific methods
    c = args.thresholdCut

    # Quadric Edge Collapse Threshold to avoid shape destruction
    number_face_collapse = 15

    # Tx and Rx positions (Modern City)
    tx = np.array([8.38372, -35.8423, 14]) 
    rx = np.array([-10.8001, 9.67042, 1.5])
    
    # Modern City coordinates
    lim_sup_esq = (-119.5, 91)
    lim_sup_dir = (75.5, 91)
    lim_inf_esq = (-119.5, -106)
    lim_inf_dir = (75.5, -106)

    
    gd = 19.5  

    # Path to original meshes/xml
    cwd = os.getcwd()
    original_meshes_path = cwd + "/mitsubas/modern_city/meshes"
    original_xml_path = cwd + "/mitsubas/modern_city/export.xml"

    # Debug
    print(f"Meshes Directory: {original_meshes_path}", end="\n\n")

    # Main Loop
    for i in range(2):
        # grid x and y
        gd = 19.5 * (i+1)
        # 19.5 * 1 >> 10x10 >> 100 total grids (19.5 x 19.5 each grid)
        # 19.5 * 2 = 39 >> 5x5 >>> 25 total grids (39 x 39 each grid)
        # 125 Total simulations

        for j in range(10/(i+1)):
            # Grid config (X)
            grid_x_start_stop = (j*gd, (j+1)*gd)
            for k in range(10/(i+1)):
                # Grid config (Y)
                grid_y_start_stop = (k*gd, (k+1)*gd)
                # Create a meshset
                ms = pymeshlab.pmeshlab.MeshSet()
                # New paths
                new_folder_path = cwd + "/mitsubas/simplified_scenarios/scenario" + i + j
                new_xml_path = cwd + "/mitsubas/simplified_scenarios/scenario" + i + j + "/export.xml"
                new_meshes_path = cwd + "/mitsubas/simplified_scenarios/scenario" + i + j + "/meshes"
                simplification_process(ms)
