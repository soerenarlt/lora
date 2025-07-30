import numpy as np
import time
import json
# -----------------------------------------------------------------------------

from data_main.graphdata import generate_graph, build_state_string, simplify_state_string
from modes_main import generate_state
import traceback

# --------------------------------------------------------------------
# LOAD MODEL

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

MODEL_ID = "Meta-Llama-3-8B"

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", load_in_8bit=True)
tok  = AutoTokenizer.from_pretrained(MODEL_ID)

model = PeftModel.from_pretrained(base, "out/checkpoint-4000")
model.eval()

# --------------------------------------------------------------------

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
job_id = int(os.environ.get("SLURM_JOB_ID", 0))

for i in range(10000):

    #generate the dicke states 
    strings_list = []
    states_list = []
    max_n = 5

    mode = task_id

    for ii, numvert in zip(range(3,8), iter([4,6,8,10,12])):
        print(f'generating state for {numvert} vertices')

        state = generate_state(ii, numvert, mode=mode)

        states_list.append(state)

        # print(state)
        if numvert <= 8:
            statestring = build_state_string(state)
            simple_statestring = simplify_state_string(statestring)
            print(statestring)
            strings_list.append(statestring)

        state.normalize()
        

    states = '|'.join(strings_list)
    
    prompt = '<|begin_of_text|><|start_header_id|>{quantum state}<|end_header_id|>'+states+'<|start_header_id|>{code}<|end_header_id|>'
    tt = time.time()
    output = model.generate(**tok(prompt, return_tensors="pt").to(0), max_new_tokens=2**13)[0]
    print(f"Time taken: {time.time() - tt:.2f} seconds")
    dec_output = tok.decode(output, skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {dec_output}")
    # pred is everything after the string '{code}'
    pred = dec_output.split('{code}')[1]


    try:
        fidelities = np.zeros(max_n)
        for N in range(max_n):
            print(f'N = {N}')
            #gg_pred is pytheus Graph, has useful methods for computing perfect matchings / fidelities etc. 
            gg_pred = generate_graph(pred,N)
            gg_pred.state.normalize()
            fidelity = (gg_pred.state@states_list[N])**2
            print(f'fidelity = {fidelity}')
            fidelities[N] = fidelity
        #if all fidelities are >0.99 then we have a perfect match
        if np.all(fidelities > 0.99):
            print('PERFECT MATCH!!!')

        #print boolean array of fidelities
        print(fidelities > 0.99)

        output_file = f"sample_results/{job_id}_{task_id}.csv"
        codes_file = f"sample_results/{job_id}_{task_id}_codes.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs(os.path.dirname(codes_file), exist_ok=True)
        with open(output_file, 'a') as f:
            f.write(f"{fidelities[0]},{fidelities[1]},{fidelities[2]},{fidelities[3]},{fidelities[4]}\n")
        with open(codes_file, "a", encoding="utf‑8") as out:
            out.write(json.dumps({"code": pred}) + "\n")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print('failed to generate valid states')
        print(pred)
        continue
    print('------------------')


