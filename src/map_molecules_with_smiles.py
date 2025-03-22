import numpy as np
import pandas as pd
import pubchempy as pcp
import json

def get_smiles(chemical_name):
    try:
        compounds = pcp.get_compounds(chemical_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
        else:
            return "Chemical name not found"
    except Exception as e:
        return str(e)

def map_mols_with_smiles():
    df_mn = pd.read_excel(r'data/SMILES.xlsx')
    molnames = np.unique([i.strip() for i in df_mn.iloc[:,:3]['API'].tolist()+df_mn.iloc[:,:3]['Coformer'].tolist()])

    molname_dictionary = {}
    for i in range(len(molnames)):
        molname_dictionary[molnames[i]] = get_smiles(molnames[i])
        print(f'{round(((i+1)/len(molnames))*100,2)}% completed...')

    print('\n'.join([i for i in molname_dictionary if molname_dictionary[i]=='Chemical name not found']))

    with open('data/molname_dictionary.txt','w') as f:
        molname_dictionary = f.write(json.dumps(molname_dictionary))

if __name__ == '__main__':
    map_mols_with_smiles()