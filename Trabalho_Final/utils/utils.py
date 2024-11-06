import numpy as np

def remove_lines_from_xml(new_xml_path: str, keyword: str) -> None:
    with open(new_xml_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Store the lines to be removed
    modified_lines = []
    skip_lines = 0

    for i in range(len(lines)):
        # If there is three instead two lines, cut them all
        #if "boolean" in lines[i] and skip_lines > 0:
        #    skip_lines += 1

        if skip_lines > 0:
            skip_lines -= 1
            continue
        
        if keyword in lines[i]:
            skip_lines = 2  # Jump 2 lines
            # if it isn't the first line, remove the previous line
            if i > 0:
                modified_lines.pop()
        else:
            modified_lines.append(lines[i])
    # Save the modified lines
    with open(new_xml_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)

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