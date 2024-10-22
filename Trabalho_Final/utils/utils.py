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
