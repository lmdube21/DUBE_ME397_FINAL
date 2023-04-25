import streamlit as st
import pandas as pd

st.write('Historical MLS Best XI Evaluation')
st.write(' ')
year = st.selectbox(
     'What Year is of Interest?',
    ('1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
       '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021', '2022'))

position = st.selectbox('What Position is of Interest?',
                        ('Forward', 'Midfield', 'Defense', 'Goalkeeper'))

model_field = pd.read_pickle('final_data/model_results_field.pkl')
true_field = pd.read_pickle('final_data/best_11_results_field.pkl')
model_gk = pd.read_pickle('final_data/model_results_gk.pkl')
true_gk = pd.read_pickle('final_data/best_11_gk.pkl')

position_dict = {
    'Forward':'F', 
    'Midfield':'M', 
    'Defense':'D', 
    'Goalkeeper':'G'
}

if position == 'Goalkeeper':
    model_df = model_gk
    true_df = true_gk
else: 
    model_df = model_field
    true_df = true_field

st.write('Top 3 Predictied Best XI Candidates:')
st.write(model_df[(model_df['POS'] == position_dict[position]) & (model_df['Year'] == int(year))])
st.write(' ')
st.write(' ')
st.write('True Best XI at Position:')
st.write(true_df[(true_df['POS'] == position_dict[position]) & (true_df['Year'] == int(year))])



