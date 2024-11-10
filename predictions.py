import numpy as np
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure, Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet
import os

# Variables
dir = r"directory"
m3gnet_model_path = r"m3gnetweights"
ehull_path = r"ehull_paths"
mp_api_key = "m3gnet_apikey"

print(f"CIF directory: {dir}")
print(f"M3GNet model path: {m3gnet_model_path}")
print(f"Energy hull prediction path: {ehull_path}")
print(f"Materials Project API key: {mp_api_key}")

def predict_ehull(dir, model_path, output_path, api_key):
    m3gnet_e_form = M3GNet.from_dir(model_path)
    ehull_list = []

    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory not found: {dir}")

    for i, file_name in enumerate(os.listdir(dir)):
        file_path = os.path.join(dir, file_name)
        print(f"Processing file {i+1}/{len(os.listdir(dir))}: {file_path}")
        
        try:
            crystal = Structure.from_file(file_path, sort=True, merge_tol=0.01)
            e_form_predict = m3gnet_e_form.predict_structure(crystal)
            print(f"Predicted formation energy for {file_name}: {e_form_predict}")
        except Exception as e:
            print(f"Could not predict formation energy of {file_name}: {e}")
            ehull_list.append((file_name, "N/A"))
            # Save intermediate results in case of crash
            np.save(output_path, np.array(ehull_list))
            continue

        elements = ''.join([i for i in crystal.formula if not i.isdigit()]).split(" ")
        print(f"Elements extracted for {file_name}: {elements}")

        try:
            mpr = MPRester(api_key)
            all_compounds = mpr.summary.search(elements=elements)
            print(f"Number of compounds returned for {elements}: {len(all_compounds)}")

            insert_list = []
            for compound in all_compounds:
                # Ensure `compound` has a `composition` attribute
                if hasattr(compound, 'composition') and hasattr(compound, 'formation_energy_per_atom'):
                    comp = str(compound.composition.reduced_composition).replace(" ", "")
                    insert_list.append(comp)
                else:
                    print(f"Skipping invalid compound for {file_name}: {compound}")
                    continue

            # Prepare the phase diagram entries
            pde_list = []
            for compound in all_compounds:
                if hasattr(compound, 'composition') and hasattr(compound, 'formation_energy_per_atom'):
                    comp = str(compound.composition.reduced_composition).replace(" ", "")
                    pde_list.append(ComputedEntry(comp, compound.formation_energy_per_atom))
                else:
                    print(f"Skipping invalid compound: {compound}")

            # Check if we have valid phase diagram data
            if not pde_list:
                raise ValueError(f"No valid phase diagram data for {file_name}")

            print(f"pde_list for {file_name}: {[entry.composition for entry in pde_list]}")

            diagram = PhaseDiagram(pde_list)
            _, pmg_ehull = diagram.get_decomp_and_e_above_hull(ComputedEntry(Composition(crystal.formula.replace(" ", "")), e_form_predict[0][0].numpy()))
            ehull_list.append((file_name, pmg_ehull))
        except Exception as e:
            print(f"Could not create phase diagram for {file_name}: {e}")
            ehull_list.append((file_name, "N/A"))

        # Save intermediate results in case of crash
        np.save(output_path, np.array(ehull_list))

    # Final save at the end
    np.save(output_path, np.array(ehull_list))

def main():
    predict_ehull(dir, m3gnet_model_path, ehull_path, mp_api_key)

if __name__ == "__main__":
    main()
