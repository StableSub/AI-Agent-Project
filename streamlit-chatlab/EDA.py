import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def run_eda(df: pd.DataFrame):
    st.subheader("기본 정보")
    st.write("데이터 크기 : ", df.shape)
    
    st.subheader("결측치 확인")
    st.write(df.isnull().sum())
    
    st.subheader("통계 요약")
    st.dataframe(df.describe())