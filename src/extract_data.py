import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import json
from matplotlib import pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

import requests
from bs4 import BeautifulSoup
import re
import time

import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


swiss_input_coformers = r"""B(C1=CC=C(C=C1)C=O)(O)O
C(=CC(=O)O)C(=O)O
C(=O)(C(=O)O)O
C(=O)(N)N
C(=S)(N)N
C(C(=O)N)C(=O)N
C(C(=O)N)O
C(C(=O)O)C(=O)O
C(C(=O)O)C(CC(=O)O)(C(=O)O)O
C(C(=O)O)N
C(C(=O)O)O
C(C(C(=O)O)N)C(=O)N
C(C(C(=O)O)N)C(=O)O
C(C(C(=O)O)O)(C(=O)O)O
C(C(C(=O)O)O)C(=O)O
C(C(C(C(=O)O)O)C(=O)O)C(=O)O
C(C(C(C(C(CO)O)O)O)O)O
C(C(C1C(=C(C(=O)O1)O)O)O)O
C(C(CO)(CO)N)O
C(C1C(C(C(C(O1)O)O)O)O)O
C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O
C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O
C(C1C(C(C(C(O1)OC2C(OC(C2O)(CO)O)CO)O)O)O)O
C(CC(=O)N)C(=O)N
C(CC(=O)N)C(C(=O)O)N
C(CC(=O)O)C(=O)C(=O)O
C(CC(=O)O)C(=O)O
C(CC(=O)O)C(C(=O)O)N
C(CC(=O)O)CC(=O)O
C(CC(=O)O)CN
C(CC(C(=O)O)N)CN=C(N)N
C(CCC(=O)O)CC(=O)O
C(CCC(=O)O)CCC(=O)O
C(CCCC(=O)O)CCC(=O)O
C(CCCC(=O)O)CCCC(=O)O
C(CCCCC(=O)O)CCCC(=O)O
C(CCCCCC(=O)O)CCCCC(=O)O
C(CCN)CC(C(=O)O)N
C(CO)N
C(CS(=O)(=O)O)S(=O)(=O)O
C1(=C(C(=C(C(=C1F)F)F)F)F)C(=O)O
C1(=C(C(=O)C(=C(C1=O)Br)O)Br)O
C1(=NC(=NC(=N1)N)N)N
C1(C(C(OC(C1O)O)C(=O)O)O)O
C1=C(C(=C(C(=C1F)F)F)F)C(=O)O
C1=C(C(=CC(=C1C#N)C#N)C#N)C#N
C1=C(C=C(C(=C1C(=O)O)O)[N+](=O)[O-])[N+](=O)[O-]
C1=C(C=C(C(=C1F)F)F)C(=O)O
C1=C(C=C(C(=C1O)C(=O)O)O)O
C1=C(C=C(C(=C1O)O)O)C(=O)O
C1=C(C=C(C(=C1O)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O
C1=C(C=C(C=C1C(=O)O)C(=O)O)C(=O)O
C1=C(C=C(C=C1C(=O)O)[N+](=O)[O-])C(=O)O
C1=C(C=C(C=C1O)O)C(=O)O
C1=C(C=C(C=C1O)O)O
C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])C(=O)O
C1=C(C=NC(=C1Br)N)Br
C1=C(C=NN1)Br
C1=C(C=NN1)I
C1=C(N=C(N=C1Cl)N)N
C1=C(NC(=O)N=C1)N
C1=C(NC=N1)CC(C(=O)O)N
C1=C(NN=C1C(=O)O)C(=O)O
C1=CC(=C(C(=C1)F)F)C(=O)O
C1=CC(=C(C(=C1)O)C(=O)O)O
C1=CC(=C(C(=C1)O)O)C(=O)O
C1=CC(=C(C(=C1)O)O)O
C1=CC(=C(C(=C1)[N+](=O)[O-])C(=O)O)C(=O)O
C1=CC(=C(C=C1C(=O)O)O)O
C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])Cl
C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])F
C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])N
C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])O
C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O
C1=CC(=C(C=C1C=CC(=O)O)O)O
C1=CC(=C(C=C1CCC(=O)O)O)O
C1=CC(=C(C=C1Cl)C(=O)O)O
C1=CC(=C(C=C1F)C(=O)O)F
C1=CC(=C(C=C1F)C(=O)O)O
C1=CC(=C(C=C1N)O)C(=O)O
C1=CC(=C(C=C1N)[N+](=O)[O-])F
C1=CC(=C(C=C1O)C(=O)O)O
C1=CC(=C(C=C1O)O)C(=O)O
C1=CC(=C(C=C1[N+](=O)[O-])C(=O)O)F
C1=CC(=C(C=C1[N+](=O)[O-])Cl)C(=O)O
C1=CC(=C(C=C1[N+](=O)[O-])Cl)N
C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O
C1=CC(=C(C=C1[N+](=O)[O-])N)Cl
C1=CC(=CC(=C1)C#N)C#N
C1=CC(=CC(=C1)C(=O)O)C(=O)O
C1=CC(=CC(=C1)Cl)C(=O)O
C1=CC(=CC(=C1)N)C(=O)N
C1=CC(=CC(=C1)N)C(=O)O
C1=CC(=CC(=C1)O)C#N
C1=CC(=CC(=C1)O)C(=O)N
C1=CC(=CC(=C1)O)C(=O)O
C1=CC(=CC(=C1)O)O
C1=CC(=CC(=C1)[N+](=O)[O-])C(=O)O
C1=CC(=CC=C1C#N)C#N
C1=CC(=CC=C1C#N)O
C1=CC(=CC=C1C(=O)N)N
C1=CC(=CC=C1C(=O)N)O
C1=CC(=CC=C1C(=O)N)[N+](=O)[O-]
C1=CC(=CC=C1C(=O)O)C(=O)O
C1=CC(=CC=C1C(=O)O)F
C1=CC(=CC=C1C(=O)O)N
C1=CC(=CC=C1C(=O)O)O
C1=CC(=CC=C1C(=O)O)[N+](=O)[O-]
C1=CC(=CC=C1C2=CC=C(C=C2)O)O
C1=CC(=CC=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O
C1=CC(=CC=C1C=CC(=O)O)O
C1=CC(=CC=C1CC(C(=O)O)N)O
C1=CC(=CC=C1CCC(=O)O)O
C1=CC(=CC=C1N)S(=O)(=O)C2=CC=C(C=C2)N
C1=CC(=CC=C1N)[N+](=O)[O-]
C1=CC(=CC=C1O)O
C1=CC(=CC=C1[N+](=O)[O-])O
C1=CC(=CN=C1)C#N
C1=CC(=CN=C1)C(=O)N
C1=CC(=CN=C1)C(=O)O
C1=CC(=CN=C1)N
C1=CC(=CN=C1)O
C1=CC(=NC(=C1)C(=O)O)C(=O)O
C1=CC(=NC(=C1)N)N
C1=CC(=NC=C1Cl)N
C1=CC(=O)NC=C1
C1=CC2=C(C=CC(=C2)O)C=C1C(=O)O
C1=CC2=C(C=CN=C2)C(=C1)O
C1=CC2=NNN=C2C=C1
C1=CC=C(C(=C1)C(=O)N)N
C1=CC=C(C(=C1)C(=O)N)O
C1=CC=C(C(=C1)C(=O)O)F
C1=CC=C(C(=C1)C(=O)O)N
C1=CC=C(C(=C1)C(=O)O)O
C1=CC=C(C(=C1)O)O
C1=CC=C(C=C1)C(=O)N
C1=CC=C(C=C1)C(=O)NCC(=O)O
C1=CC=C(C=C1)C(=O)O
C1=CC=C(C=C1)C(C(=O)O)O
C1=CC=C(C=C1)C(CC(=O)O)C(=O)O
C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O
C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)CCC(=O)O
C1=CC=C(C=C1)C2=CC=CC=C2O
C1=CC=C(C=C1)C2=CC=NC=C2
C1=CC=C(C=C1)C2=NC3=C(N=C(N=C3N=C2N)N)N
C1=CC=C(C=C1)C=CC(=O)O
C1=CC=C(C=C1)CC(C(=O)O)N
C1=CC=C(C=C1)CCC(=O)O
C1=CC=C(C=C1)SCCC(=O)O
C1=CC=C2C(=C1)C(=CN2)C(=O)O
C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N
C1=CC=C2C(=C1)C(=NO2)CS(=O)(=O)N
C1=CC=C2C(=C1)C(=O)NS2(=O)=O
C1=CC=C2C(=C1)C=C(C(=C2CC3=C(C(=CC4=CC=CC=C43)C(=O)O)O)O)C(=O)O
C1=CC=C2C(=C1)C=C(N2)C(=O)O
C1=CC=C2C(=C1)C=CC(=C2O)C(=O)O
C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N
C1=CC=C2C(=C1)C=CC=C2C#N
C1=CC=C2C(=C1)C=CC=C2O
C1=CC=C2C(=C1)C=CC=C2S(=O)(=O)O
C1=CC=C2C(=C1)C=CN2
C1=CC=C2C(=C1)N=C3C=CC=CC3=N2
C1=CC=C2C(=C1)NC=N2
C1=CC=C2C=C(C(=CC2=C1)C(=O)O)O
C1=CC=C2C=C3C=CC=CC3=CC2=C1
C1=CC=C2C=CC=CC2=C1
C1=CC=NC(=C1)C(=O)N
C1=CC=NC(=C1)C(=O)O
C1=CC=NC(=C1)C2=CC=CC=N2
C1=CC=NC(=C1)N
C1=CN=C(C(=N1)C(=O)O)C(=O)O
C1=CN=C(C=C1C(=O)O)C(=O)O
C1=CN=C(C=N1)C(=O)N
C1=CN=C(C=N1)C(=O)O
C1=CN=C(N=C1)Cl
C1=CN=C(N=C1)N
C1=CN=CC=C1C#N
C1=CN=CC=C1C(=O)N
C1=CN=CC=C1C(=O)NN
C1=CN=CC=C1C(=O)O
C1=CN=CC=C1C2=CC=NC=C2
C1=CN=CC=C1C=CC2=CC=NC=C2
C1=CN=CC=C1CCC2=CC=NC=C2
C1=CN=CC=C1CCCC2=CC=NC=C2
C1=CN=CC=C1N
C1=CN=CC=C1N=NC2=CC=NC=C2
C1=CN=CC=N1
C1=CN=CN1
C1=CNC(=O)C(=C1)C(=O)O
C1=CNN=C1
C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl
C1=C[N+](=CC=C1[N+](=O)[O-])[O-]
C1=NC(=O)NC(=C1F)N
C1=NC2=NC=NC(=C2N1)N
C1=NC=NN1
C1C(C(C(C(O1)(CO)O)O)O)O
C1C(C(C(C(O1)O)O)O)O
C1C2=CC=CC=C2COC1=O
C1C2CC3CC1CC(C2)(C3)C(=O)O
C1CC(=O)N(C1)CC(=O)N
C1CC(=O)NC(=O)C1
C1CC(=O)NC1C(=O)O
C1CC(CCC1N)N
C1CC(NC1)C(=O)O
C1CC1C2=CC=C(C3=CC=CC=C23)N4C(=NN=C4Br)SCC(=O)O
C1CCC(=O)NCC1
C1CCC(CC1)NS(=O)(=O)O
C1CCNC(=O)C1
C1CCNCC1
C1CN2CCN1CC2
C1CNCCN1
C1CNCCNCCCNCCNC1
C1COCCN1
C=CCCCCCCCCC(=O)O
CC(=CC1=CC=CC=C1)C=C2C(=O)N(C(=S)S2)CC(=O)O
CC(=O)C(=O)O
CC(=O)CCCCN1C(=O)C2=C(N=CN2C)N(C1=O)C
CC(=O)N
CC(=O)NC(CS)C(=O)O
CC(=O)NC1=CC=C(C=C1)C(=O)O
CC(=O)NC1=CC=C(C=C1)O
CC(=O)NC1=NN=C(S1)S(=O)(=O)N
CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC
CC(=O)NS(=O)(=O)C1=CC=C(C=C1)N
CC(=O)OC1=CC=CC=C1C(=O)O
CC(C(=O)N)O
CC(C(=O)O)N
CC(C(=O)O)O
CC(C(C(=O)O)N)O
CC(C(C)C(=O)O)C(=O)O
CC(C)(C)C1=C(C=CC(=C1)O)O
CC(C)(C)CCNC(CC(=O)O)C(=O)NC(CC1=CC=CC=C1)C(=O)OC
CC(C)(CC(=O)O)C(=O)O
CC(C)CC(C(=O)O)N
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
CC(C)OC1=CC=C(C=C1)C(=O)NS(=O)(=O)C2=CC=C(C=C2)N
CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O
CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O
CC(CCCC(=O)O)CC(=O)O
CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O
CC(O)(P(=O)(O)O)P(=O)(O)O
CC1(C(CCC1(C)C(=O)O)C(=O)O)C
CC1=C(C(=NN1)C)Cl
CC1=C(C(=NN1)C)I
CC1=C(C(=O)C=CO1)O
CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O
CC1=C(C=C(C=C1)C(C)C)O
CC1=C(C=CC=C1Cl)NC2=CC=CC=C2C(=O)O
CC1=C(N=C(C(=N1)C)C)C
CC1=C(SC(=N1)C2=CC(=C(C=C2)OCC(C)C)C#N)C(=O)O
CC1=CC(=C(C=C1)N)C(=O)O
CC1=CC(=CC(=C1)O)O
CC1=CC(=NC(=N1)N)C
CC1=CC(=NC(=N1)N)Cl
CC1=CC(=NC(=N1)NS(=O)(=O)C2=CC=C(C=C2)N)C
CC1=CC(=NN1)C
CC1=CC(=O)NC(=N1)N
CC1=CC(=O)NC=C1
CC1=CC=C(C=C1)S(=O)(=O)O
CC1=CC=CC(=O)N1
CC1=CC=CC=C1C(=O)O
CC1=CNC(=O)NC1=O
CC1=CNC2=CC=CC=C12
CC1=NC=CN1
CC=CC=CC(=O)O
CCC(C(=O)O)C(=O)O
CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C
CCC1=C(C(=O)C=CO1)O
CCCC(=O)O
CCCCCCCC(=O)O
CCCCCCCCCCCCCCCCCC(=O)O
CCCCCCCCCCCCCCCCCCN
CCCCCCCCCCCCCCCCO
CCCCCCCCNCC(C(C(C(CO)O)O)O)O
CCCOC(=O)C1=CC(=C(C(=C1)O)O)O
CCN(CC)CC(=O)NC1=C(C=CC=C1C)C
CCOC1=CC=CC=C1C(=O)N
CN(C)C1=CC=NC=C1
CN1C(=O)N2C=NC(=C2N=N1)C(=O)N
CN1C2=C(C(=O)N(C1=O)C)NC=N2
CN1C2=C(C(=O)N(C1=O)C)NC=N2.O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
CN1C=NC2=C1C(=O)NC(=O)N2C
CN1CCCC1=O
CN1CCN(CC1)C
CN1CCN(CC1)CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)Cl
CN1CCOCC1
CNCC(=O)O
CNCC(C(C(C(CO)O)O)O)O
COC(=O)C1=CC(=C(C(=C1)O)O)O
COC(=O)C1=CC=C(C=C1)O
COC(=O)NC1C(C(C(OC1OC2C(OC(C(C2O)N)OC3C(OC(C(C3O)N)O)CO)CO)CO)OC4C(C(C(C(O4)CO)OC5C(C(C(C(O5)CO)OC6C(C(C(C(O6)CO)OC7C(C(C(C(O7)CO)OC8C(C(C(C(O8)CO)OC9C(C(C(C(O9)CO)O)O)N)O)N)O)N)O)N)O)N)O)N)O
COC1=C(C=C(C=C1)C2CC(=O)C3=C(C=C(C=C3O2)O)O)O
COC1=C(C=C(C=C1)[N+](=O)[O-])N
COC1=C(C=CC(=C1)C(=O)O)O
COC1=C(C=CC(=C1)C2C(OC3=C(O2)C=C(C=C3)C4C(C(=O)C5=C(C=C(C=C5O4)O)O)O)CO)O
COC1=C(C=CC(=C1)C=CC(=O)O)O
COC1=C(C=CC(=C1)C=O)O
COC1=CC(=CC(=C1)C=CC2=CC=C(C=C2)O)OC
COC1=CC(=CC(=C1O)OC)C(=O)O
COC1=CC(=CC(=C1O)OC)C=CC(=O)O
COC1=CC(=CC(=C1OC)OC)C(=O)O
C[N+](=O)[O-]
C[N+](C)(C)CC(=O)[O-]
NS(=O)(=O)O
OS(=O)(=O)O"""

swiss_input = r"""C(C(=O)O)C(CC(=O)O)(C(=O)O)O
C(C(C(=O)O)O)(C(=O)O)O
C(C(C(=O)O)O)C(=O)O
C(CC(=O)O)C(=O)O
C(CC(=O)O)CC(=O)O
C(CCC(=O)O)CC(=O)O
C(CCC(=O)O)CCC(=O)O
C(CCCC(=O)O)CCC(=O)O
C(CCCC(=O)O)CCCC(=O)O
C(CCCCC(=O)O)CCCC(=O)O
C(CCCCCC(=O)O)CCCCC(=O)O
C1=C(C(=O)NC(=O)N1)F
C1=C(C=C(C(=C1O)O)O)C2C(C(=O)C3=C(C=C(C=C3O2)O)O)O
C1=C(N=C(C(=N1)N)Br)Br
C1=C(N=CC(=N1)Br)N
C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O
C1=CC(=C(C=C1C2=C(C=C(C=C2)F)F)C(=O)O)O
C1=CC(=C(C=C1CC(=O)O)O)O
C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O
C1=CC(=C(C=C1N)O)C(=O)O
C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O
C1=CC(=CC(=C1)O)C#N
C1=CC(=CC=C1C#N)O
C1=CC(=CC=C1C(=O)N)N
C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O
C1=CC(=CC=C1C2=COC3=C(C2=O)C=CC(=C3)O)O
C1=CC(=CC=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O
C1=CC(=CC=C1C=CC(=O)C2=C(C=C(C=C2)O)O)O
C1=CC(=CC=C1C=CC2=CC(=CC(=C2)O)O)O
C1=CC(=CC=C1CCC(=O)C2=C(C=C(C=C2O)O)O)O
C1=CC(=CN=C1)C#N
C1=CC(=CN=C1)C(=O)N
C1=CC(=CN=C1)C(=O)NCCO[N+](=O)[O-]
C1=CC2=C(C=C1OC(F)(F)F)SC(=N2)N
C1=CC=C(C(=C1)C(=O)O)NC2=CC=CC(=C2)C(F)(F)F
C1=CC=C(C(=C1)C(=O)O)O
C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl
C1=CC=C(C(=C1)CC(=O)OCC(=O)O)NC2=C(C=CC=C2Cl)Cl
C1=CC=C(C=C1)C(=O)N
C1=CC=C(C=C1)C(=O)O
C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O
C1=CC=C(C=C1)CN2C=CN=C2C3=NC=CN3CC4=CC=CC=C4
C1=CC=C2C(=C1)C=CC(=O)O2
C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N
C1=CC=C2C=CC=CC2=C1
C1=CN=C(C=N1)C(=O)N
C1=CN=CC=C1C#N
C1=CN=CC=C1C(=O)NN
C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl
C1=NC(=C2C(=N1)N(C=N2)CCOCP(=O)(O)O)N
C1=NC2=C(N1COCCO)N=C(NC2=O)N
C1C(OC2=CC(=CC(=C2C1=O)O)O)C3=CC=C(C=C3)O
C1CC(=O)N(C1)CC(=O)N
C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N
C1CC(=O)NC1=O
C1CC(OC1)N2C=C(C(=O)NC2=O)F
C1CC2=C(C=CC(=C2)F)OC1C(CNCC(C3CCC4=C(O3)C=CC(=C4)F)O)O.Cl
C1CCC(CC1)(CC(=O)O)CN
C1CCC(CC1)C(=O)N2CC3C4=CC=CC=C4CCN3C(=O)C2
C1CCN(CC1)C(=O)C=CC=CC2=CC3=C(C=C2)OCO3
C1CSSC1CCCCC(=O)O
C1NC2=CC(=C(C=C2S(=O)(=O)N1)S(=O)(=O)N)Cl
CC(=CCC1=C(C(=C(C=C1O)OC)C(=O)C=CC2=CC=C(C=C2)O)O)C
CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C
CC(=O)NC1=CC=C(C=C1)O
CC(=O)NC1=NN=C(S1)S(=O)(=O)N
CC(=O)OC1=CC=CC=C1C(=O)NC2=NC=C(S2)[N+](=O)[O-]
CC(=O)OC1=CC=CC=C1C(=O)O
CC(C)(C)C(=O)OCOP(=O)(COCCN1C=NC2=C(N=CN=C21)N)OCOC(=O)C(C)(C)C
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
CC(C)N1C(=CC=N1)C2=C(C=CC=N2)COC3=CC=CC(=C3C=O)O
CC(C)OC(=O)C(C)(C)OC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)Cl
CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O
CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O
CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O
CC(C1=CC2=C(C=C1)SC3=CC=CC=C3C(=O)C2)C(=O)O
CC(C1=NC=NC=C1F)C(CN2C=NC=N2)(C3=C(C=C(C=C3)F)F)O
CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O
CC1(CCC(C23C1C(C(C45C2CCC(C4O)C(=C)C5=O)(OC3)O)O)O)C
CC12CCC3C(C1CCC2(C#C)O)CCC4=CC5=C(CC34C)C=NO5
CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C
CC1=C(C(=C(C(=C1O)OC)OC)O)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C
CC1=C(C(=CC=C1)NC2=CC=CC=C2C(=O)O)C
CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C(C)C
CC1=C(C(CC(C1)O)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CC(CC2(C)C)O)C)C)C
CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O
CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)OCC(=O)O
CC1=C(C=C(C(=O)N1)C#N)C2=CC=NC=C2
CC1=C(N=C(C(=N1)C)C)C
CC1=C2C=C(C=CC2=C(C(=N1)C(=O)NCC(=O)O)O)OC3=CC=CC=C3
CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
CC1=CN=C(S1)NC(=O)C2=C(C3=CC=CC=C3S(=O)(=O)N2C)O
CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C
CC1=NC=C(N1CC(C)O)[N+](=O)[O-]
CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O
CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=CC(=C4C(=O)CC(OC4=C3)C5=CC(=C(C=C5)OC)O)O)O)O)O)O)O)O
CC1CC(=O)C=C(C12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)OC
CCC(C(=O)N)N1CCCC1=O
CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C
CCC(C)N1C(=O)N(C=N1)C2=CC=C(C=C2)N3CCN(CC3)C4=CC=C(C=C4)OCC5COC(O5)(CN6C=NC=N6)C7=C(C=C(C=C7)Cl)Cl
CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4
CCCCCCCCCCCCCCCCCC(=O)O
CCCN(CCC)S(=O)(=O)C1=CC=C(C=C1)C(=O)O
CCN1C=C(C(=O)C2=C1N=C(C=C2)C)C(=O)O
CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O
CCN1CCCC1CNC(=O)C2=C(C=CC(=C2)S(=O)(=O)N)OC
CCOC(=O)N1CCC(=C2C3=C(CCC4=C2N=CC=C4)C=C(C=C3)Cl)CC1
CCOC1=CC=CC=C1C(=O)N
CN1C(=C(C2=CC=CC=C2S1(=O)=O)O)C(=O)NC3=CC=CC=N3
CN1C(=O)N2C=NC(=C2N=N1)C(=O)N
CN1C2=C(C(=O)N(C1=O)C)NC=N2
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=C3C(=CC(=C4)F)N2
CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F.Cl
COC(=O)C=CC(=O)OC
COC1=C(C=C(C=C1)Cl)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3
COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4
COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O
COC1=C(C=CC(=C1)C=CC(=O)O)O
COC1=CC(=CC(=C1O)OC)C=CC(=O)O"""


def get_dataset():

    df_mn = pd.read_excel(r'data/SMILES.xlsx')
    molnames = np.unique([i.strip() for i in df_mn.iloc[:,:3]['API'].tolist()+df_mn.iloc[:,:3]['Coformer'].tolist()])

    with open('data/molname_dictionary.txt','r') as f:
        molname_dictionary = json.loads(f.read())

    final_molnames = np.unique([i for i in molnames if (i in molname_dictionary and molname_dictionary[i]!='Chemical name not found')])
    final_molnames_lower_unique = np.unique([i.lower() for i in final_molnames])

    molname_dictionary_withoutduplicates = {}
    for i in molname_dictionary:
        if i.lower() not in molname_dictionary_withoutduplicates:
            if i.lower() in final_molnames_lower_unique:
                molname_dictionary_withoutduplicates[i.lower()] = molname_dictionary[i]

    mol_list = []
    for i in final_molnames_lower_unique:
        mol_list.append(molname_dictionary_withoutduplicates[i.lower()])

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in mol_list]

    df_u = calc.pandas(mols)

    df_u['mol_name'] = mol_list

    all_values = []
    for i in range(len(df_mn)):
        api = df_mn.iloc[i,0].strip()
        coformer = df_mn.iloc[i,1].strip()
        output_raw = df_mn.iloc[i,2]
        if api.lower() in molname_dictionary_withoutduplicates and coformer.lower() in molname_dictionary_withoutduplicates:
            api_values = df_u.loc[df_u.iloc[:,-1]==molname_dictionary_withoutduplicates[api.lower()],:].values
            coformer_values = df_u.loc[df_u.iloc[:,-1]==molname_dictionary_withoutduplicates[coformer.lower()],:].values
            if 'yes' in output_raw.lower():
                output = 1
            elif 'no' in output_raw.lower():
                output = 0
            else:
                print('some problem occurred.')
            all_values.append([api_values,coformer_values,output])
        else:
            print('Either api or coformer not found.', 'API=',api,'. Coformer=',coformer)

    df = pd.read_csv('data/swissadme_karthick.csv')
    df['Original Canonical SMILES'] = swiss_input.splitlines()

    df_coformers = pd.read_csv('data/swissadme_coformer.csv')
    df_coformers['Original Canonical SMILES'] = swiss_input_coformers.splitlines()

    all_smiles_list = list(molname_dictionary_withoutduplicates.values())
    all_smiles_list_without_duplicates = np.unique(all_smiles_list)
    smiles_dictionary = {}

    calc2 = Calculator(descriptors, ignore_3D=True)
    mols2 = [Chem.MolFromSmiles(smi) for smi in all_smiles_list_without_duplicates]
    df_unique2 = calc.pandas(mols2)

    smiles_dictionary = dict(zip(all_smiles_list_without_duplicates,df_unique2.values.tolist()))

    all_values = []
    strings_of_dtypes = [str(i) for i in df_unique2.dtypes.values]
    columns_to_be_shortlisted = [j for j in range(len(strings_of_dtypes)) if 'int' in strings_of_dtypes[j] or 'float' in strings_of_dtypes[j]]
    api_and_coformer_smiles = []
    found = 0
    notfound_count = 0
    for i in range(len(df_mn)):
        api = df_mn.iloc[i,0].strip()
        coformer = df_mn.iloc[i,1].strip()
        output_raw = df_mn.iloc[i,2].strip()
        if output_raw.lower() == 'yes':
            output = 1
        elif output_raw.lower() == 'no':
            output = 0
        try:
            extracted_api_values = np.array(smiles_dictionary[molname_dictionary_withoutduplicates[api.lower()]])[columns_to_be_shortlisted]
            api_values = np.array(extracted_api_values,dtype=float)
            api_found_index = np.where(df.iloc[:,-1].str.lower()==molname_dictionary_withoutduplicates[api.lower()].lower().strip())
            if 0 not in api_found_index[0].shape:
                extra_api_values = df.iloc[api_found_index[0][0],[True if ('int' in str(i) or 'float' in str(i)) else False for i in df.dtypes.values]].values
                api_values = np.r_[api_values,extra_api_values]
                found += 1
            else:
                print('API', api, 'not found!')
                notfound_count += 1
        except Exception as e:
            print(e)
            continue
        try:
            extracted_coformer_values = np.array(smiles_dictionary[molname_dictionary_withoutduplicates[coformer.lower()]])[columns_to_be_shortlisted]
            coformer_values = np.array(extracted_coformer_values,dtype=float)
            coformer_found_index = np.where(df_coformers.iloc[:,-1].str.lower()==molname_dictionary_withoutduplicates[coformer.lower()].lower().strip())
            if 0 not in coformer_found_index[0].shape:
                extra_coformer_values = df_coformers.iloc[api_found_index[0][0],[True if ('int' in str(i) or 'float' in str(i)) else False for i in df_coformers.dtypes.values]].values
                coformer_values = np.r_[coformer_values,extra_coformer_values]
                found += 1
            else:
                print('Coformer', coformer, 'not found!')
                notfound_count += 1
        except Exception as e:
            print(e)
            continue
        all_values.append([api_values,coformer_values,output]+[api,coformer])
        all_values.append([coformer_values,api_values,output]+[coformer,api]) # Data augmentation
        api_and_coformer_smiles.append([api,coformer])

    x_data = []
    y_data = []
    for i in range(len(all_values)):
        x_data.append(all_values[i][0].tolist()+all_values[i][1].tolist()+[all_values[i][3],all_values[i][4]])
        y_data.append([all_values[i][2]]+[all_values[i][3],all_values[i][4]])

    x_data = pd.DataFrame(x_data)
    y_data = pd.DataFrame(y_data)

    return x_data, y_data

# def get_dataset():

#     # map_mols_with_smiles()

#     df_mn = pd.read_excel(r'data/SMILES.xlsx')
#     molnames = np.unique([i.strip() for i in df_mn.iloc[:,:3]['API'].tolist()+df_mn.iloc[:,:3]['Coformer'].tolist()])

#     with open('data/molname_dictionary.txt','r') as f:
#         molname_dictionary = json.loads(f.read())

#     final_molnames = np.unique([i for i in molnames if (i in molname_dictionary and molname_dictionary[i]!='Chemical name not found')])
#     final_molnames_lower_unique = np.unique([i.lower() for i in final_molnames])

#     molname_dictionary_withoutduplicates = {}
#     for i in molname_dictionary:
#         if i.lower() not in molname_dictionary_withoutduplicates:
#             if i.lower() in final_molnames_lower_unique:
#                 molname_dictionary_withoutduplicates[i.lower()] = molname_dictionary[i]

#     mol_list = []
#     # for i in final_molnames:
#     for i in final_molnames_lower_unique:
#         #mol_list.append(molname_dictionary[i])
#         mol_list.append(molname_dictionary_withoutduplicates[i.lower()])



#     calc = Calculator(descriptors, ignore_3D=True)
#     mols = [Chem.MolFromSmiles(smi) for smi in mol_list]

#     df_u = calc.pandas(mols)

#     df_u['mol_name'] = mol_list

#     all_values = []
#     for i in range(len(df_mn)):
#         api = df_mn.iloc[i,0].strip()
#         coformer = df_mn.iloc[i,1].strip()
#         output_raw = df_mn.iloc[i,2]
#         if api.lower() in molname_dictionary_withoutduplicates and coformer.lower() in molname_dictionary_withoutduplicates:
#             api_values = df_u.loc[df_u.iloc[:,-1]==molname_dictionary_withoutduplicates[api.lower()],:].values
#             coformer_values = df_u.loc[df_u.iloc[:,-1]==molname_dictionary_withoutduplicates[coformer.lower()],:].values
#             if 'yes' in output_raw.lower():
#                 output = 1
#             elif 'no' in output_raw.lower():
#                 output = 0
#             else:
#                 print('some problem occurred.')
#             all_values.append([api_values,coformer_values,output])
#         else:
#             print('Either api or coformer not found.', 'API=',api,'. Coformer=',coformer)



#     swiss_input_coformers = r"""B(C1=CC=C(C=C1)C=O)(O)O
#     C(=CC(=O)O)C(=O)O
#     C(=O)(C(=O)O)O
#     C(=O)(N)N
#     C(=S)(N)N
#     C(C(=O)N)C(=O)N
#     C(C(=O)N)O
#     C(C(=O)O)C(=O)O
#     C(C(=O)O)C(CC(=O)O)(C(=O)O)O
#     C(C(=O)O)N
#     C(C(=O)O)O
#     C(C(C(=O)O)N)C(=O)N
#     C(C(C(=O)O)N)C(=O)O
#     C(C(C(=O)O)O)(C(=O)O)O
#     C(C(C(=O)O)O)C(=O)O
#     C(C(C(C(=O)O)O)C(=O)O)C(=O)O
#     C(C(C(C(C(CO)O)O)O)O)O
#     C(C(C1C(=C(C(=O)O1)O)O)O)O
#     C(C(CO)(CO)N)O
#     C(C1C(C(C(C(O1)O)O)O)O)O
#     C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O
#     C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O
#     C(C1C(C(C(C(O1)OC2C(OC(C2O)(CO)O)CO)O)O)O)O
#     C(CC(=O)N)C(=O)N
#     C(CC(=O)N)C(C(=O)O)N
#     C(CC(=O)O)C(=O)C(=O)O
#     C(CC(=O)O)C(=O)O
#     C(CC(=O)O)C(C(=O)O)N
#     C(CC(=O)O)CC(=O)O
#     C(CC(=O)O)CN
#     C(CC(C(=O)O)N)CN=C(N)N
#     C(CCC(=O)O)CC(=O)O
#     C(CCC(=O)O)CCC(=O)O
#     C(CCCC(=O)O)CCC(=O)O
#     C(CCCC(=O)O)CCCC(=O)O
#     C(CCCCC(=O)O)CCCC(=O)O
#     C(CCCCCC(=O)O)CCCCC(=O)O
#     C(CCN)CC(C(=O)O)N
#     C(CO)N
#     C(CS(=O)(=O)O)S(=O)(=O)O
#     C1(=C(C(=C(C(=C1F)F)F)F)F)C(=O)O
#     C1(=C(C(=O)C(=C(C1=O)Br)O)Br)O
#     C1(=NC(=NC(=N1)N)N)N
#     C1(C(C(OC(C1O)O)C(=O)O)O)O
#     C1=C(C(=C(C(=C1F)F)F)F)C(=O)O
#     C1=C(C(=CC(=C1C#N)C#N)C#N)C#N
#     C1=C(C=C(C(=C1C(=O)O)O)[N+](=O)[O-])[N+](=O)[O-]
#     C1=C(C=C(C(=C1F)F)F)C(=O)O
#     C1=C(C=C(C(=C1O)C(=O)O)O)O
#     C1=C(C=C(C(=C1O)O)O)C(=O)O
#     C1=C(C=C(C(=C1O)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O
#     C1=C(C=C(C=C1C(=O)O)C(=O)O)C(=O)O
#     C1=C(C=C(C=C1C(=O)O)[N+](=O)[O-])C(=O)O
#     C1=C(C=C(C=C1O)O)C(=O)O
#     C1=C(C=C(C=C1O)O)O
#     C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])C(=O)O
#     C1=C(C=NC(=C1Br)N)Br
#     C1=C(C=NN1)Br
#     C1=C(C=NN1)I
#     C1=C(N=C(N=C1Cl)N)N
#     C1=C(NC(=O)N=C1)N
#     C1=C(NC=N1)CC(C(=O)O)N
#     C1=C(NN=C1C(=O)O)C(=O)O
#     C1=CC(=C(C(=C1)F)F)C(=O)O
#     C1=CC(=C(C(=C1)O)C(=O)O)O
#     C1=CC(=C(C(=C1)O)O)C(=O)O
#     C1=CC(=C(C(=C1)O)O)O
#     C1=CC(=C(C(=C1)[N+](=O)[O-])C(=O)O)C(=O)O
#     C1=CC(=C(C=C1C(=O)O)O)O
#     C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])Cl
#     C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])F
#     C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])N
#     C1=CC(=C(C=C1C(=O)O)[N+](=O)[O-])O
#     C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O
#     C1=CC(=C(C=C1C=CC(=O)O)O)O
#     C1=CC(=C(C=C1CCC(=O)O)O)O
#     C1=CC(=C(C=C1Cl)C(=O)O)O
#     C1=CC(=C(C=C1F)C(=O)O)F
#     C1=CC(=C(C=C1F)C(=O)O)O
#     C1=CC(=C(C=C1N)O)C(=O)O
#     C1=CC(=C(C=C1N)[N+](=O)[O-])F
#     C1=CC(=C(C=C1O)C(=O)O)O
#     C1=CC(=C(C=C1O)O)C(=O)O
#     C1=CC(=C(C=C1[N+](=O)[O-])C(=O)O)F
#     C1=CC(=C(C=C1[N+](=O)[O-])Cl)C(=O)O
#     C1=CC(=C(C=C1[N+](=O)[O-])Cl)N
#     C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O
#     C1=CC(=C(C=C1[N+](=O)[O-])N)Cl
#     C1=CC(=CC(=C1)C#N)C#N
#     C1=CC(=CC(=C1)C(=O)O)C(=O)O
#     C1=CC(=CC(=C1)Cl)C(=O)O
#     C1=CC(=CC(=C1)N)C(=O)N
#     C1=CC(=CC(=C1)N)C(=O)O
#     C1=CC(=CC(=C1)O)C#N
#     C1=CC(=CC(=C1)O)C(=O)N
#     C1=CC(=CC(=C1)O)C(=O)O
#     C1=CC(=CC(=C1)O)O
#     C1=CC(=CC(=C1)[N+](=O)[O-])C(=O)O
#     C1=CC(=CC=C1C#N)C#N
#     C1=CC(=CC=C1C#N)O
#     C1=CC(=CC=C1C(=O)N)N
#     C1=CC(=CC=C1C(=O)N)O
#     C1=CC(=CC=C1C(=O)N)[N+](=O)[O-]
#     C1=CC(=CC=C1C(=O)O)C(=O)O
#     C1=CC(=CC=C1C(=O)O)F
#     C1=CC(=CC=C1C(=O)O)N
#     C1=CC(=CC=C1C(=O)O)O
#     C1=CC(=CC=C1C(=O)O)[N+](=O)[O-]
#     C1=CC(=CC=C1C2=CC=C(C=C2)O)O
#     C1=CC(=CC=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O
#     C1=CC(=CC=C1C=CC(=O)O)O
#     C1=CC(=CC=C1CC(C(=O)O)N)O
#     C1=CC(=CC=C1CCC(=O)O)O
#     C1=CC(=CC=C1N)S(=O)(=O)C2=CC=C(C=C2)N
#     C1=CC(=CC=C1N)[N+](=O)[O-]
#     C1=CC(=CC=C1O)O
#     C1=CC(=CC=C1[N+](=O)[O-])O
#     C1=CC(=CN=C1)C#N
#     C1=CC(=CN=C1)C(=O)N
#     C1=CC(=CN=C1)C(=O)O
#     C1=CC(=CN=C1)N
#     C1=CC(=CN=C1)O
#     C1=CC(=NC(=C1)C(=O)O)C(=O)O
#     C1=CC(=NC(=C1)N)N
#     C1=CC(=NC=C1Cl)N
#     C1=CC(=O)NC=C1
#     C1=CC2=C(C=CC(=C2)O)C=C1C(=O)O
#     C1=CC2=C(C=CN=C2)C(=C1)O
#     C1=CC2=NNN=C2C=C1
#     C1=CC=C(C(=C1)C(=O)N)N
#     C1=CC=C(C(=C1)C(=O)N)O
#     C1=CC=C(C(=C1)C(=O)O)F
#     C1=CC=C(C(=C1)C(=O)O)N
#     C1=CC=C(C(=C1)C(=O)O)O
#     C1=CC=C(C(=C1)O)O
#     C1=CC=C(C=C1)C(=O)N
#     C1=CC=C(C=C1)C(=O)NCC(=O)O
#     C1=CC=C(C=C1)C(=O)O
#     C1=CC=C(C=C1)C(C(=O)O)O
#     C1=CC=C(C=C1)C(CC(=O)O)C(=O)O
#     C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O
#     C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)CCC(=O)O
#     C1=CC=C(C=C1)C2=CC=CC=C2O
#     C1=CC=C(C=C1)C2=CC=NC=C2
#     C1=CC=C(C=C1)C2=NC3=C(N=C(N=C3N=C2N)N)N
#     C1=CC=C(C=C1)C=CC(=O)O
#     C1=CC=C(C=C1)CC(C(=O)O)N
#     C1=CC=C(C=C1)CCC(=O)O
#     C1=CC=C(C=C1)SCCC(=O)O
#     C1=CC=C2C(=C1)C(=CN2)C(=O)O
#     C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N
#     C1=CC=C2C(=C1)C(=NO2)CS(=O)(=O)N
#     C1=CC=C2C(=C1)C(=O)NS2(=O)=O
#     C1=CC=C2C(=C1)C=C(C(=C2CC3=C(C(=CC4=CC=CC=C43)C(=O)O)O)O)C(=O)O
#     C1=CC=C2C(=C1)C=C(N2)C(=O)O
#     C1=CC=C2C(=C1)C=CC(=C2O)C(=O)O
#     C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N
#     C1=CC=C2C(=C1)C=CC=C2C#N
#     C1=CC=C2C(=C1)C=CC=C2O
#     C1=CC=C2C(=C1)C=CC=C2S(=O)(=O)O
#     C1=CC=C2C(=C1)C=CN2
#     C1=CC=C2C(=C1)N=C3C=CC=CC3=N2
#     C1=CC=C2C(=C1)NC=N2
#     C1=CC=C2C=C(C(=CC2=C1)C(=O)O)O
#     C1=CC=C2C=C3C=CC=CC3=CC2=C1
#     C1=CC=C2C=CC=CC2=C1
#     C1=CC=NC(=C1)C(=O)N
#     C1=CC=NC(=C1)C(=O)O
#     C1=CC=NC(=C1)C2=CC=CC=N2
#     C1=CC=NC(=C1)N
#     C1=CN=C(C(=N1)C(=O)O)C(=O)O
#     C1=CN=C(C=C1C(=O)O)C(=O)O
#     C1=CN=C(C=N1)C(=O)N
#     C1=CN=C(C=N1)C(=O)O
#     C1=CN=C(N=C1)Cl
#     C1=CN=C(N=C1)N
#     C1=CN=CC=C1C#N
#     C1=CN=CC=C1C(=O)N
#     C1=CN=CC=C1C(=O)NN
#     C1=CN=CC=C1C(=O)O
#     C1=CN=CC=C1C2=CC=NC=C2
#     C1=CN=CC=C1C=CC2=CC=NC=C2
#     C1=CN=CC=C1CCC2=CC=NC=C2
#     C1=CN=CC=C1CCCC2=CC=NC=C2
#     C1=CN=CC=C1N
#     C1=CN=CC=C1N=NC2=CC=NC=C2
#     C1=CN=CC=N1
#     C1=CN=CN1
#     C1=CNC(=O)C(=C1)C(=O)O
#     C1=CNN=C1
#     C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl
#     C1=C[N+](=CC=C1[N+](=O)[O-])[O-]
#     C1=NC(=O)NC(=C1F)N
#     C1=NC2=NC=NC(=C2N1)N
#     C1=NC=NN1
#     C1C(C(C(C(O1)(CO)O)O)O)O
#     C1C(C(C(C(O1)O)O)O)O
#     C1C2=CC=CC=C2COC1=O
#     C1C2CC3CC1CC(C2)(C3)C(=O)O
#     C1CC(=O)N(C1)CC(=O)N
#     C1CC(=O)NC(=O)C1
#     C1CC(=O)NC1C(=O)O
#     C1CC(CCC1N)N
#     C1CC(NC1)C(=O)O
#     C1CC1C2=CC=C(C3=CC=CC=C23)N4C(=NN=C4Br)SCC(=O)O
#     C1CCC(=O)NCC1
#     C1CCC(CC1)NS(=O)(=O)O
#     C1CCNC(=O)C1
#     C1CCNCC1
#     C1CN2CCN1CC2
#     C1CNCCN1
#     C1CNCCNCCCNCCNC1
#     C1COCCN1
#     C=CCCCCCCCCC(=O)O
#     CC(=CC1=CC=CC=C1)C=C2C(=O)N(C(=S)S2)CC(=O)O
#     CC(=O)C(=O)O
#     CC(=O)CCCCN1C(=O)C2=C(N=CN2C)N(C1=O)C
#     CC(=O)N
#     CC(=O)NC(CS)C(=O)O
#     CC(=O)NC1=CC=C(C=C1)C(=O)O
#     CC(=O)NC1=CC=C(C=C1)O
#     CC(=O)NC1=NN=C(S1)S(=O)(=O)N
#     CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC
#     CC(=O)NS(=O)(=O)C1=CC=C(C=C1)N
#     CC(=O)OC1=CC=CC=C1C(=O)O
#     CC(C(=O)N)O
#     CC(C(=O)O)N
#     CC(C(=O)O)O
#     CC(C(C(=O)O)N)O
#     CC(C(C)C(=O)O)C(=O)O
#     CC(C)(C)C1=C(C=CC(=C1)O)O
#     CC(C)(C)CCNC(CC(=O)O)C(=O)NC(CC1=CC=CC=C1)C(=O)OC
#     CC(C)(CC(=O)O)C(=O)O
#     CC(C)CC(C(=O)O)N
#     CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
#     CC(C)OC1=CC=C(C=C1)C(=O)NS(=O)(=O)C2=CC=C(C=C2)N
#     CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O
#     CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O
#     CC(CCCC(=O)O)CC(=O)O
#     CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O
#     CC(O)(P(=O)(O)O)P(=O)(O)O
#     CC1(C(CCC1(C)C(=O)O)C(=O)O)C
#     CC1=C(C(=NN1)C)Cl
#     CC1=C(C(=NN1)C)I
#     CC1=C(C(=O)C=CO1)O
#     CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O
#     CC1=C(C=C(C=C1)C(C)C)O
#     CC1=C(C=CC=C1Cl)NC2=CC=CC=C2C(=O)O
#     CC1=C(N=C(C(=N1)C)C)C
#     CC1=C(SC(=N1)C2=CC(=C(C=C2)OCC(C)C)C#N)C(=O)O
#     CC1=CC(=C(C=C1)N)C(=O)O
#     CC1=CC(=CC(=C1)O)O
#     CC1=CC(=NC(=N1)N)C
#     CC1=CC(=NC(=N1)N)Cl
#     CC1=CC(=NC(=N1)NS(=O)(=O)C2=CC=C(C=C2)N)C
#     CC1=CC(=NN1)C
#     CC1=CC(=O)NC(=N1)N
#     CC1=CC(=O)NC=C1
#     CC1=CC=C(C=C1)S(=O)(=O)O
#     CC1=CC=CC(=O)N1
#     CC1=CC=CC=C1C(=O)O
#     CC1=CNC(=O)NC1=O
#     CC1=CNC2=CC=CC=C12
#     CC1=NC=CN1
#     CC=CC=CC(=O)O
#     CCC(C(=O)O)C(=O)O
#     CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C
#     CCC1=C(C(=O)C=CO1)O
#     CCCC(=O)O
#     CCCCCCCC(=O)O
#     CCCCCCCCCCCCCCCCCC(=O)O
#     CCCCCCCCCCCCCCCCCCN
#     CCCCCCCCCCCCCCCCO
#     CCCCCCCCNCC(C(C(C(CO)O)O)O)O
#     CCCOC(=O)C1=CC(=C(C(=C1)O)O)O
#     CCN(CC)CC(=O)NC1=C(C=CC=C1C)C
#     CCOC1=CC=CC=C1C(=O)N
#     CN(C)C1=CC=NC=C1
#     CN1C(=O)N2C=NC(=C2N=N1)C(=O)N
#     CN1C2=C(C(=O)N(C1=O)C)NC=N2
#     CN1C2=C(C(=O)N(C1=O)C)NC=N2.O
#     CN1C=NC2=C1C(=O)N(C(=O)N2C)C
#     CN1C=NC2=C1C(=O)NC(=O)N2C
#     CN1CCCC1=O
#     CN1CCN(CC1)C
#     CN1CCN(CC1)CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)Cl
#     CN1CCOCC1
#     CNCC(=O)O
#     CNCC(C(C(C(CO)O)O)O)O
#     COC(=O)C1=CC(=C(C(=C1)O)O)O
#     COC(=O)C1=CC=C(C=C1)O
#     COC(=O)NC1C(C(C(OC1OC2C(OC(C(C2O)N)OC3C(OC(C(C3O)N)O)CO)CO)CO)OC4C(C(C(C(O4)CO)OC5C(C(C(C(O5)CO)OC6C(C(C(C(O6)CO)OC7C(C(C(C(O7)CO)OC8C(C(C(C(O8)CO)OC9C(C(C(C(O9)CO)O)O)N)O)N)O)N)O)N)O)N)O)N)O
#     COC1=C(C=C(C=C1)C2CC(=O)C3=C(C=C(C=C3O2)O)O)O
#     COC1=C(C=C(C=C1)[N+](=O)[O-])N
#     COC1=C(C=CC(=C1)C(=O)O)O
#     COC1=C(C=CC(=C1)C2C(OC3=C(O2)C=C(C=C3)C4C(C(=O)C5=C(C=C(C=C5O4)O)O)O)CO)O
#     COC1=C(C=CC(=C1)C=CC(=O)O)O
#     COC1=C(C=CC(=C1)C=O)O
#     COC1=CC(=CC(=C1)C=CC2=CC=C(C=C2)O)OC
#     COC1=CC(=CC(=C1O)OC)C(=O)O
#     COC1=CC(=CC(=C1O)OC)C=CC(=O)O
#     COC1=CC(=CC(=C1OC)OC)C(=O)O
#     C[N+](=O)[O-]
#     C[N+](C)(C)CC(=O)[O-]
#     NS(=O)(=O)O
#     OS(=O)(=O)O"""


#     swiss_input = r"""C(C(=O)O)C(CC(=O)O)(C(=O)O)O
#     C(C(C(=O)O)O)(C(=O)O)O
#     C(C(C(=O)O)O)C(=O)O
#     C(CC(=O)O)C(=O)O
#     C(CC(=O)O)CC(=O)O
#     C(CCC(=O)O)CC(=O)O
#     C(CCC(=O)O)CCC(=O)O
#     C(CCCC(=O)O)CCC(=O)O
#     C(CCCC(=O)O)CCCC(=O)O
#     C(CCCCC(=O)O)CCCC(=O)O
#     C(CCCCCC(=O)O)CCCCC(=O)O
#     C1=C(C(=O)NC(=O)N1)F
#     C1=C(C=C(C(=C1O)O)O)C2C(C(=O)C3=C(C=C(C=C3O2)O)O)O
#     C1=C(N=C(C(=N1)N)Br)Br
#     C1=C(N=CC(=N1)Br)N
#     C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O
#     C1=CC(=C(C=C1C2=C(C=C(C=C2)F)F)C(=O)O)O
#     C1=CC(=C(C=C1CC(=O)O)O)O
#     C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O
#     C1=CC(=C(C=C1N)O)C(=O)O
#     C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O
#     C1=CC(=CC(=C1)O)C#N
#     C1=CC(=CC=C1C#N)O
#     C1=CC(=CC=C1C(=O)N)N
#     C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O
#     C1=CC(=CC=C1C2=COC3=C(C2=O)C=CC(=C3)O)O
#     C1=CC(=CC=C1C2=COC3=CC(=CC(=C3C2=O)O)O)O
#     C1=CC(=CC=C1C=CC(=O)C2=C(C=C(C=C2)O)O)O
#     C1=CC(=CC=C1C=CC2=CC(=CC(=C2)O)O)O
#     C1=CC(=CC=C1CCC(=O)C2=C(C=C(C=C2O)O)O)O
#     C1=CC(=CN=C1)C#N
#     C1=CC(=CN=C1)C(=O)N
#     C1=CC(=CN=C1)C(=O)NCCO[N+](=O)[O-]
#     C1=CC2=C(C=C1OC(F)(F)F)SC(=N2)N
#     C1=CC=C(C(=C1)C(=O)O)NC2=CC=CC(=C2)C(F)(F)F
#     C1=CC=C(C(=C1)C(=O)O)O
#     C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl
#     C1=CC=C(C(=C1)CC(=O)OCC(=O)O)NC2=C(C=CC=C2Cl)Cl
#     C1=CC=C(C=C1)C(=O)N
#     C1=CC=C(C=C1)C(=O)O
#     C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O
#     C1=CC=C(C=C1)CN2C=CN=C2C3=NC=CN3CC4=CC=CC=C4
#     C1=CC=C2C(=C1)C=CC(=O)O2
#     C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N
#     C1=CC=C2C=CC=CC2=C1
#     C1=CN=C(C=N1)C(=O)N
#     C1=CN=CC=C1C#N
#     C1=CN=CC=C1C(=O)NN
#     C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl
#     C1=NC(=C2C(=N1)N(C=N2)CCOCP(=O)(O)O)N
#     C1=NC2=C(N1COCCO)N=C(NC2=O)N
#     C1C(OC2=CC(=CC(=C2C1=O)O)O)C3=CC=C(C=C3)O
#     C1CC(=O)N(C1)CC(=O)N
#     C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N
#     C1CC(=O)NC1=O
#     C1CC(OC1)N2C=C(C(=O)NC2=O)F
#     C1CC2=C(C=CC(=C2)F)OC1C(CNCC(C3CCC4=C(O3)C=CC(=C4)F)O)O.Cl
#     C1CCC(CC1)(CC(=O)O)CN
#     C1CCC(CC1)C(=O)N2CC3C4=CC=CC=C4CCN3C(=O)C2
#     C1CCN(CC1)C(=O)C=CC=CC2=CC3=C(C=C2)OCO3
#     C1CSSC1CCCCC(=O)O
#     C1NC2=CC(=C(C=C2S(=O)(=O)N1)S(=O)(=O)N)Cl
#     CC(=CCC1=C(C(=C(C=C1O)OC)C(=O)C=CC2=CC=C(C=C2)O)O)C
#     CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C
#     CC(=O)NC1=CC=C(C=C1)O
#     CC(=O)NC1=NN=C(S1)S(=O)(=O)N
#     CC(=O)OC1=CC=CC=C1C(=O)NC2=NC=C(S2)[N+](=O)[O-]
#     CC(=O)OC1=CC=CC=C1C(=O)O
#     CC(C)(C)C(=O)OCOP(=O)(COCCN1C=NC2=C(N=CN=C21)N)OCOC(=O)C(C)(C)C
#     CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
#     CC(C)N1C(=CC=N1)C2=C(C=CC=N2)COC3=CC=CC(=C3C=O)O
#     CC(C)OC(=O)C(C)(C)OC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)Cl
#     CC(C1=CC(=C(C=C1)C2=CC=CC=C2)F)C(=O)O
#     CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O
#     CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O
#     CC(C1=CC2=C(C=C1)SC3=CC=CC=C3C(=O)C2)C(=O)O
#     CC(C1=NC=NC=C1F)C(CN2C=NC=N2)(C3=C(C=C(C=C3)F)F)O
#     CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O
#     CC1(CCC(C23C1C(C(C45C2CCC(C4O)C(=C)C5=O)(OC3)O)O)O)C
#     CC12CCC3C(C1CCC2(C#C)O)CCC4=CC5=C(CC34C)C=NO5
#     CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C
#     CC1=C(C(=C(C(=C1O)OC)OC)O)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C
#     CC1=C(C(=CC=C1)NC2=CC=CC=C2C(=O)O)C
#     CC1=C(C(=O)N(N1C)C2=CC=CC=C2)C(C)C
#     CC1=C(C(CC(C1)O)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CC(CC2(C)C)O)C)C)C
#     CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O
#     CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)OCC(=O)O
#     CC1=C(C=C(C(=O)N1)C#N)C2=CC=NC=C2
#     CC1=C(N=C(C(=N1)C)C)C
#     CC1=C2C=C(C=CC2=C(C(=N1)C(=O)NCC(=O)O)O)OC3=CC=CC=C3
#     CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
#     CC1=CN=C(S1)NC(=O)C2=C(C3=CC=CC=C3S(=O)(=O)N2C)O
#     CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C
#     CC1=NC=C(N1CC(C)O)[N+](=O)[O-]
#     CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=C(OC4=CC(=CC(=C4C3=O)O)O)C5=CC(=C(C=C5)O)O)O)O)O)O)O)O
#     CC1C(C(C(C(O1)OCC2C(C(C(C(O2)OC3=CC(=C4C(=O)CC(OC4=C3)C5=CC(=C(C=C5)OC)O)O)O)O)O)O)O)O
#     CC1CC(=O)C=C(C12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)OC
#     CCC(C(=O)N)N1CCCC1=O
#     CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C
#     CCC(C)N1C(=O)N(C=N1)C2=CC=C(C=C2)N3CCN(CC3)C4=CC=C(C=C4)OCC5COC(O5)(CN6C=NC=N6)C7=C(C=C(C=C7)Cl)Cl
#     CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4
#     CCCCCCCCCCCCCCCCCC(=O)O
#     CCCN(CCC)S(=O)(=O)C1=CC=C(C=C1)C(=O)O
#     CCN1C=C(C(=O)C2=C1N=C(C=C2)C)C(=O)O
#     CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O
#     CCN1CCCC1CNC(=O)C2=C(C=CC(=C2)S(=O)(=O)N)OC
#     CCOC(=O)N1CCC(=C2C3=C(CCC4=C2N=CC=C4)C=C(C=C3)Cl)CC1
#     CCOC1=CC=CC=C1C(=O)N
#     CN1C(=C(C2=CC=CC=C2S1(=O)=O)O)C(=O)NC3=CC=CC=N3
#     CN1C(=O)N2C=NC(=C2N=N1)C(=O)N
#     CN1C2=C(C(=O)N(C1=O)C)NC=N2
#     CN1C=NC2=C1C(=O)N(C(=O)N2C)C
#     CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=C3C(=CC(=C4)F)N2
#     CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F.Cl
#     COC(=O)C=CC(=O)OC
#     COC1=C(C=C(C=C1)Cl)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3
#     COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4
#     COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O
#     COC1=C(C=CC(=C1)C=CC(=O)O)O
#     COC1=CC(=CC(=C1O)OC)C=CC(=O)O"""


#     df = pd.read_csv('data/swissadme_karthick.csv')
#     df['Original Canonical SMILES'] = swiss_input.splitlines()

#     df_coformers = pd.read_csv('data/swissadme_coformer.csv')
#     df_coformers['Original Canonical SMILES'] = swiss_input_coformers.splitlines()


#     all_smiles_list = list(molname_dictionary_withoutduplicates.values())
#     all_smiles_list_without_duplicates = np.unique(all_smiles_list)
#     smiles_dictionary = {}

#     calc2 = Calculator(descriptors, ignore_3D=True)
#     mols2 = [Chem.MolFromSmiles(smi) for smi in all_smiles_list_without_duplicates]
#     df_unique2 = calc.pandas(mols2)

#     smiles_dictionary = dict(zip(all_smiles_list_without_duplicates,df_unique2.values.tolist()))

#     all_values = []
#     strings_of_dtypes = [str(i) for i in df_unique2.dtypes.values]
#     columns_to_be_shortlisted = [j for j in range(len(strings_of_dtypes)) if 'int' in strings_of_dtypes[j] or 'float' in strings_of_dtypes[j]]
#     api_and_coformer_smiles = []
#     found = 0
#     notfound_count = 0
#     for i in range(len(df_mn)):
#         api = df_mn.iloc[i,0].strip()
#         coformer = df_mn.iloc[i,1].strip()
#         output_raw = df_mn.iloc[i,2].strip()
#         if output_raw.lower() == 'yes':
#             output = 1
#         elif output_raw.lower() == 'no':
#             output = 0
#         try:
#             extracted_api_values = np.array(smiles_dictionary[molname_dictionary_withoutduplicates[api.lower()]])[columns_to_be_shortlisted]
#             api_values = np.array(extracted_api_values,dtype=float)
#             api_found_index = np.where(df.iloc[:,-1].str.lower()==molname_dictionary_withoutduplicates[api.lower()].lower().strip())
#             if 0 not in api_found_index[0].shape:
#                 extra_api_values = df.iloc[api_found_index[0][0],[True if ('int' in str(i) or 'float' in str(i)) else False for i in df.dtypes.values]].values
#                 api_values = np.r_[api_values,extra_api_values]
#                 found += 1
#             else:
#                 print('API', api, 'not found!')
#                 notfound_count += 1
#         except Exception as e:
#             print(e)
#             continue
#         try:
#             extracted_coformer_values = np.array(smiles_dictionary[molname_dictionary_withoutduplicates[coformer.lower()]])[columns_to_be_shortlisted]
#             coformer_values = np.array(extracted_coformer_values,dtype=float)
#             coformer_found_index = np.where(df_coformers.iloc[:,-1].str.lower()==molname_dictionary_withoutduplicates[coformer.lower()].lower().strip())
#             if 0 not in coformer_found_index[0].shape:
#                 extra_coformer_values = df_coformers.iloc[api_found_index[0][0],[True if ('int' in str(i) or 'float' in str(i)) else False for i in df_coformers.dtypes.values]].values
#                 coformer_values = np.r_[coformer_values,extra_coformer_values]
#                 found += 1
#             else:
#                 print('Coformer', coformer, 'not found!')
#                 notfound_count += 1
#         except Exception as e:
#             print(e)
#             continue
#         all_values.append([api_values,coformer_values,output]+[api,coformer])
#         all_values.append([coformer_values,api_values,output]+[coformer,api]) # Data augmentation
#         api_and_coformer_smiles.append([api,coformer])


#         x_data = []
#     y_data = []
#     for i in range(len(all_values)):
#         x_data.append(all_values[i][0].tolist()+all_values[i][1].tolist()+[all_values[i][3],all_values[i][4]])
#         y_data.append([all_values[i][2]]+[all_values[i][3],all_values[i][4]])

#     x_data = pd.DataFrame(x_data)
#     y_data = pd.DataFrame(y_data)

#     return x_data, y_data
