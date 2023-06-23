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
