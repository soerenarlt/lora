import json
import glob
import pathlib
import os

import h5py
import numpy as np

token_dict = json.load(open('tok.json', 'r'))

def detokenize_indices(indices):
    # Create a reverse mapping from indices to tokens
    reverse_dict = {index: token for token, index in token_dict.items()}
    
    #remove padding tokens
    indices = [index for index in indices if index != token_dict['<PAD>']]

    # Convert the list of indices to the corresponding tokens
    output_str = ''.join(reverse_dict.get(index, '') for index in indices)
    
    return output_str



if __name__ == "__main__":
    for i in range(99):

        with h5py.File(f'data/split_data_{i}.h5', 'r') as f:
            
            num_samples = len(f['code'])
            out_path = f'data/processed_data_{i}.json'
            
            with open(out_path, "w", encoding="utf‑8") as out:
                for i in range(num_samples):
                    code_inds = f['code'][i]
                    state_inds = f['state'][i]


                    # Detokenize the indices
                    code_str1 = detokenize_indices(code_inds)[5:-5]
                    state_str2 = detokenize_indices(state_inds)[5:-5]


                    print(code_str1)
                    print(state_str2)

                    llama_string = '<|begin_of_text|>'+'<|start_header_id|>{quantum state}<|end_header_id|>'+state_str2+'<|start_header_id|>{code}<|end_header_id|>'+code_str1+'<|end_of_text|>'
                    out.write(json.dumps({"text": llama_string}) + "\n")

