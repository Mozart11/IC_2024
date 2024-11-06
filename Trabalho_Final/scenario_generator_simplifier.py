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

# Tensorflow and gpu parameters
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpu_num = ""
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

import tensorflow as tf

class Simplification:
    def __init__(self, ms: MeshSet) -> None:
        self.ms = ms

    def simplification_algorithm(self) -> MeshSet:
        # Apply the algorithm
        if args.simplification_method == "quadric":
            n_faces = self.ms.current_mesh().face_number()
            if n_faces > number_face_collapse: # Only objects with more than 15 faces (objects with less than 15 faces are destroyed by the decimate)
                self.ms.meshing_decimation_quadric_edge_collapse(targetperc=qec_parameter)
        elif args.simplification_method == "vertex":
            self.ms.meshing_decimation_clustering(threshold=pymeshlab.pmeshlab.PercentageValue(args.parameter))
    
    def scene_generator(self, out: str, vertice_pos: np.ndarray, keyword_to_cut_xml: str) -> None:
        vertice_pos = get_center_of_vertice_pos(vertice_pos)
        # Mechanisms to divide the scenario into grids 
        if (vertice_pos[0] >= lim_sup_esq[0] + grid_x_start_stop[0] and vertice_pos[0] <= lim_sup_esq[0] + grid_x_start_stop[1]) and (vertice_pos[1] <= lim_sup_esq[1] - grid_y_start_stop[0] and vertice_pos[1] >= lim_sup_esq[1] - grid_y_start_stop[1]):
            self.simplification_algorithm()
            self.ms.save_current_mesh(out)
            return None
        else:
            # Cut 
            remove_lines_from_xml(new_xml_path, keyword_to_cut_xml)

def parse_arguments() -> argparse.Namespace:
    # Choose the simplification type
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplification_method","-m", required=False, type=str, help="Choose the algorithm to simplify (vertex, quadric)")
    parser.add_argument("--cut_type","-ct", required=True, type=str, help="Choose the type of cut (scene_generator)")
    return parser.parse_args()

def simplification_process(ms: MeshSet) -> None:
    # Adjust the output folder (xml + clean meshes)
    if not os.path.exists(new_folder_path):
    # Create scenario folder, if it doesn't exist
        os.makedirs(new_folder_path)
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
    
    # Check the argsparser
    if args.cut_type == "scene_generator":
        print("Running the Scene Generator")
        simplify_and_cut_function = simplification.scene_generator
        parameters = ["out", "vertice_pos", "keyword_to_cut_xml"]
    else:
        raise Exception("Simplification type invalid")

    for m in tqdm(os.listdir(original_meshes_path), colour="green"):
        # Load the mesh
        ms.load_new_mesh(os.path.join(original_meshes_path, m))
        cm = ms.current_mesh()
        # Position of the object
        vertice_pos = cm.vertex_matrix()
        # Save the mesh
        out = os.path.join(new_meshes_path, str(m))
        # To avoid cut the floor
        object_name = str(m)
        # String to identify objects in xml to cut
        keyword_to_cut_xml = "/" + str(m)

        # Associate strings with the real parameters
        parameters_dict = {
            "out": out,
            "vertice_pos": vertice_pos,
            "keyword_to_cut_xml": keyword_to_cut_xml,
        }
        arg = [parameters_dict[arg] for arg in parameters]

        # Simplify and cut objects
        simplify_and_cut_function(*arg)

if __name__ == "__main__":
    # Argparses
    args = parse_arguments()

    # Quadric Edge Collapse Threshold to avoid shape destruction
    number_face_collapse = 15

    # Tx and Rx positions (Modern City)
    tx = np.array([8.38372, -35.8423, 14]) 
    rx = np.array([-10.8001, 9.67042, 1.5])
    
    # Modern City coordinates
    lim_sup_esq = (-105.4, 91)
    lim_sup_dir = (78.5, 91)
    lim_inf_esq = (-105.4, -106)
    lim_inf_dir = (78.5, -106)

    # qec values
    qec_list = []

    # Path to original meshes/xml
    cwd = os.getcwd()

    i_range = 10
    # Main Loop
    for i in range(i_range):
        # grid x and y, start, stop and range
        gd = 50 + (i*10)

        # Grids size
        j_range = np.ceil((np.abs(lim_sup_esq[0]) + np.abs(lim_sup_dir[0]))/gd).astype(int)
        k_range = np.ceil((np.abs(lim_sup_esq[1]) + np.abs(lim_inf_dir[1]))/gd).astype(int)

        # Debug
        print(f"Total simulations com i: {i_range}, grid: {j_range} x {k_range}, Total simulations: {j_range * k_range}")
        print("i_range: ", i_range)
        print("j_range: ", j_range)
        print("k_range: ", k_range)

        for j in range(j_range):
            # Grid config (X) (start_stop)
            grid_x_start_stop = (j*gd, (j+1)*gd)
            for k in range(k_range):
                # Grid config (Y) (start_stop)
                grid_y_start_stop = (k*gd, (k+1)*gd)
                # Create a meshset
                ms = pymeshlab.pmeshlab.MeshSet()
                # Original Paths
                original_meshes_path = cwd + "/mitsubas/scenarios/scenario" + str(i) + str(j) + str(k) + "/meshes"
                original_xml_path = cwd +    "/mitsubas/scenarios/scenario" + str(i) + str(j) + str(k) + "/modern_export.xml"
                # New Paths
                new_folder_path = cwd + "/mitsubas/simplified_scenarios/scenario" + str(i) + str(j) + str(k)
                new_xml_path = cwd + "/mitsubas/simplified_scenarios/scenario" + str(i) + str(j) + str(k) + "/modern_export.xml"
                new_meshes_path = cwd + "/mitsubas/simplified_scenarios/scenario" + str(i) + str(j) + str(k) + "/meshes"
                
                # Debug
                print(f"Meshes Directory: {original_meshes_path}", end="\n\n")

                # QEC
                qec_parameter = np.random.rand()
                while qec_parameter < 0.25:
                    qec_parameter = np.random.rand() 
                print("QEC Parameter: ", qec_parameter)
                qec_list.append(qec_parameter)
                simplification_process(ms)

    # Save QEC parameters
    np.savez("qec_list.npz", 
            qec=qec_list)

            
            

