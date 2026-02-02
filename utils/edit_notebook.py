import json
import sys
import os

def update_cell_by_id(notebook_path, cell_id, new_source):
    if not os.path.exists(notebook_path):
        print(f"Error: File {notebook_path} not found")
        return False
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb.get('cells', []):
        if cell.get('id') == cell_id:
            # new_source should be a list of strings, each ending with \n except the last one ideally
            # but usually it's easier to pass a single string and split it
            if isinstance(new_source, str):
                lines = new_source.splitlines(keepends=True)
                # Ensure each line has a newline except maybe the last one
                cell['source'] = lines
            else:
                cell['source'] = new_source
            found = True
            break
    
    if not found:
        print(f"Error: Cell ID '{cell_id}' not found")
        return False
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully updated cell '{cell_id}'")
    return True

def add_cell_after_id(notebook_path, after_id, new_cell):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    found = False
    for cell in nb.get('cells', []):
        new_cells.append(cell)
        if cell.get('id') == after_id:
            new_cells.append(new_cell)
            found = True
    
    if not found:
        print(f"Error: Cell ID '{after_id}' not found")
        return False
    
    nb['cells'] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully added cell after '{after_id}'")
    return True

if __name__ == "__main__":
    # Example usage: python edit_notebook.py <path> <id> <source_file>
    if len(sys.argv) < 4:
        print("Usage: python edit_notebook.py <path> <id> <source_file>")
        sys.exit(1)
    
    path = sys.argv[1]
    cell_id = sys.argv[2]
    source_file = sys.argv[3]
    
    with open(source_file, 'r', encoding='utf-8') as f:
        new_source = f.read()
    
    update_cell_by_id(path, cell_id, new_source)
