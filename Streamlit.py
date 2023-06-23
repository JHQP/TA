import streamlit as st
import pandas as pd
import numpy as np
import pyrebase
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

config = {
    "apiKey": "AIzaSyBu3Lfwo_HdNGLtFFCCEXdDssgAVcG1ShA",
    "authDomain": "drinkingwaterquality-fab02.firebaseapp.com",
    "databaseURL": "https://drinkingwaterquality-fab02-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "drinkingwaterquality-fab02",
    "storageBucket": "drinkingwaterquality-fab02.appspot.com",
    "messagingSenderId": "313523960550",
    "appId": "1:313523960550:web:f9a2bbb83f7d8a6f0ad927",
    "measurementId": "G-YEDZLEWDK2"
  }

firebase = pyrebase.initialize_app(config)
database = firebase.database()

EColi = database.child("WATER QUALITY").child("MIKROBIOLOGI PARAMETERS").child("E COLI").get().val()
Coliform = database.child("WATER QUALITY").child("MIKROBIOLOGI PARAMETERS").child("COLIFORM").get().val()

Arsen = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("ARSEN").get().val()
TotalKromium = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("TOTAL KRONIUM").get().val()
Kadmium = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("KADMIUM").get().val()
Nitrit = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("NITRIT").get().val()
Nitrat = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("NITRAT").get().val()
Selenium = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("SELENIUM").get().val()
Aluminium = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("ALUMINIUM").get().val()
Sulfat = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("SULFAT").get().val()
Tembaga = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("TEMBAGA").get().val()
BOD = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("BOD").get().val()
COD = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("COD").get().val()
Amonia = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("AMONIA").get().val()
Besi = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("BESI").get().val()
SisaChlor = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("SISA CHLOR").get().val()
Sianida = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("SIANIDA").get().val()
Seng = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("SENG").get().val()
Mangan = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("MANGAN").get().val()
Klorida = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("KLORIDA").get().val()
KesadahanTotal = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("KESADAHAN TOTAL").get().val()

Bau = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("BAU").get().val()
Rasa = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("RASA").get().val()
Warna = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("WARNA").get().val()

pH = database.child("WATER QUALITY").child("CHEMICAL PARAMETERS").child("PH").get().val()
TDS = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("TDS").get().val()
Turbidity = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("TURBIDITY").get().val()
Suhu = database.child("WATER QUALITY").child("PHYSICS PARAMETERS").child("SUHU").get().val()

Uji = np.array([EColi,
        Coliform,
        Arsen,
        TotalKromium,
        Kadmium,
        Nitrit,
        Nitrat,
        Selenium,
        Aluminium,
        pH,
        Sulfat,
        Tembaga,
        TDS,
        Turbidity])

uji = Uji.astype(float)

df = pd.DataFrame(Uji)
df = df.T
df = pd.DataFrame(df, columns = ['E.Coli', 'Coliform', 'Arsen', 'Total Kromium', 'Kadmium', 'Nitrit', 'Nitrat', 'Selenium', 'Aluminium', 'pH', 'Sulfat', 'Tembaga', 'TDS', 'Turbidity'], index=[''])



data = pd.read_csv("https://raw.githubusercontent.com/JHQP/TA/main/Hasil%20Pengujian%20Kualitas%20Air%20PDAM%20-%20DataFrame.csv", index_col=0)

dfs = data.drop(['Kelayakan'], axis='columns')
dfs['E.Coli']= dfs['E.Coli'].str.replace(',','.')
dfs['Coliform']= dfs['Coliform'].str.replace(',','.')
dfs['Arsen (As)']= dfs['Arsen (As)'].str.replace(',','.')
dfs['Total Kromium (Cr-T)']= dfs['Total Kromium (Cr-T)'].str.replace(',','.')
dfs['Kadmium (Cd)']= dfs['Kadmium (Cd)'].str.replace(',','.')
dfs['Nitrit (NO2-N)']= dfs['Nitrit (NO2-N)'].str.replace(',','.')
dfs['Nitrat (NO3-N)']= dfs['Nitrat (NO3-N)'].str.replace(',','.')
dfs['Selenium (Se)']= dfs['Selenium (Se)'].str.replace(',','.')
dfs['Alumunium (Al)']= dfs['Alumunium (Al)'].str.replace(',','.')
dfs['pH']= dfs['pH'].str.replace(',','.')
dfs['Sulfat (SO4)']= dfs['Sulfat (SO4)'].str.replace(',','.')
dfs['Tembaga (Cu)']= dfs['Tembaga (Cu)'].str.replace(',','.')
dfs['Padatan Terlarut Total (TDS)']= dfs['Padatan Terlarut Total (TDS)'].str.replace(',','.')
dfs['Kekeruhan']= dfs['Kekeruhan'].str.replace(',','.')

dfs = dfs.astype(float)

dfs['Kelayakan'] = data['Kelayakan']

dfs = dfs.dropna()

X=dfs.drop(['Kelayakan'], axis='columns').to_numpy()
y=dfs['Kelayakan'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)


def elm_fit(X, target, h, W=None):
    if W is None:
        W = np.random.uniform(-.1, .1, (h, len(X[0])))

    print('Hinit')
    Hinit = X @ W.T
    print(Hinit)
    print('H')
    H = 1 / (1 + np.exp(-Hinit))
    print(H)
    Ht = H.T
    print('H+')
    Hp = np.linalg.inv(Ht @ H) @ Ht
    print(Hp)
    print('beta')
    beta = Hp @ target
    print(beta)
    print('y')
    y = H @ beta
    print(y)
    mape = sum(abs(y - target) / target) * 100 / len(target)

    return W, beta, mape

def elm_predict_test(uji, W, b, round_output=False):
    Hinit = X @ W.T
    H = 1 / (1 + np.exp(-Hinit))
    y = H @ b

    if round_output:
        y = [int(round(x)) for x in y]

    return y

W, b, mape = elm_fit(X_train, y_train, 10)
predict = elm_predict_test(uji, W, b, round_output=True)



#st.set_page_config(layout="wide")

import streamlit as st

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
  st.write('')

with col2:
  st.title('Kualitas Air Minum')

with col3:
  st.write('')

#st.sidebar.info('Test')

#st.markdown('Test')

st.write("""
<style>
.big-font {
    font-size:44px !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([.4,.6])

col1.header("KELAYAKAN :")
with col2:
  if predict[0] == 2:
    st.write('<p class="big-font">Tidak Layak</p>', unsafe_allow_html=True)

  else:
    st.write('<p class="big-font">Layak</p>', unsafe_allow_html=True)

st.divider()

st.header('Physics Parameters')
col1, col2, col3 = st.columns(3)
with col1:
  st.write('Bau :', Bau)
  st.write('TDS :', TDS)

with col2:
  st .write('Rasa :', Rasa)
  st .write('Turbidity :', Turbidity)

with col3:
  st.write('Suhu', Suhu)
  st.write('Warna', Warna)

st.divider()

st.header('Chemical Parameters')
col1, col2, col3 = st.columns(3)
with col1:
  st.write('Aluminium :', Aluminium)
  st.write('Besi :', Besi)
  st.write('Kadmium :', Kadmium)
  st.write('Mangan :', Mangan)
  st.write('pH :', pH)
  st.write('Sianida :', Sianida)
  st.write('Tembaga :', Tembaga)


with col2:
  st .write('Amonia :', Amonia)
  st .write('BOD :', BOD)
  st .write('Kesadahan Total :', KesadahanTotal)
  st .write('Nitrat :', Nitrat)
  st .write('Selenium :', Selenium)
  st .write('Sisa Chlor :', SisaChlor)
  st .write('Total Kromium :', TotalKromium)

with col3:
  st.write('Arsen :', Arsen)
  st.write('COD :', COD)
  st.write('Klorida :', Klorida)
  st.write('Nitrit :', Nitrit)
  st.write('Seng :', Seng)
  st.write('Sulfat :', Sulfat)

st.divider()

st.header('Micro-Biology Parameters')
col1, col2, col3 = st.columns(3)
with col1:
  st.write('E.Coli :', EColi)

with col2:
  st .write('Coliform :', Coliform)

with col3:
  st.write('')
