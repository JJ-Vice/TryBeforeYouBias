import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from yaml import safe_load
import user_evaluation_variables
databaseDF = None
from profanity_check import predict, predict_prob

def check_profanity(df):
    cleanedDF = df
    for i, row in cleanedDF.iterrows():
        if 'Target' in df:
            if predict([row['Target']])[0] != 0.0:
                cleanedDF.at[i, 'Target'] = '**NSFW**'
    return cleanedDF
def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections = check_profanity(df_with_selections)
    df_with_selections.insert(0, "Select", True)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)
def add_user_evalID_columns_to_df(df, evalDataPath):
    with open(evalDataPath, 'r') as f:
        yamlData = safe_load(f)
        for user in yamlData['evaluations']['username']:
            if df is None:
                df = pd.DataFrame(yamlData['evaluations']['username'][user]).T
                df.insert(0, "Eval. ID", list(yamlData['evaluations']['username'][user].keys()), True)
            else:
                df = pd.concat([df, pd.DataFrame(yamlData['evaluations']['username'][user]).T],
                                       ignore_index=True)
            evalIDIterator = 0
            for index, row in df.iterrows():
                if row['Eval. ID'] is np.nan:
                    df.loc[index, 'Eval. ID'] = list(yamlData['evaluations']['username'][user].keys())[
                        evalIDIterator]
                    evalIDIterator += 1
    return df
def initialise_page(tab):
    global databaseDF
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("\U0001F30E General Bias")
            with st.form("gen_bias_database_loading_form", clear_on_submit=False):
                communityGEN = st.form_submit_button("TBYB Community Evaluations")
                if communityGEN:
                    databaseDF = None
                    databaseDF = add_user_evalID_columns_to_df(databaseDF, './data/general_eval_database.yaml')[["Eval. ID", "Model", "Model Type", "Resolution", "No. Samples", "Inference Steps",
                             "Objects", "Actions", "Occupations", "Dist. Bias", "Hallucination", "Gen. Miss Rate",
                             "Run Time", "Date", "Time"]]
        with c2:
            st.subheader("\U0001F3AF Task-Oriented Bias")
            with st.form("task_oriented_database_loading_form", clear_on_submit=False):
                communityTASK = st.form_submit_button("TBYB Community Evaluations")
                if communityTASK:
                    databaseDF = None
                    databaseDF = add_user_evalID_columns_to_df(databaseDF, './data/task_oriented_eval_database.yaml')[["Eval. ID", "Model", "Model Type", "Resolution", "No. Samples", "Inference Steps",
                                             "Target", "Dist. Bias", "Hallucination", "Gen. Miss Rate", "Run Time", "Date", "Time"]]
        if databaseDF is not None:
            selection = dataframe_with_selections(databaseDF)
            normalised = st.toggle('Normalize Data (better for direct comparisons)')
            submitCOMPARE = st.button("Compare Selected Models")

            if submitCOMPARE:
                plot_comparison_graphs(tab, selection, normalised)

def normalise_data(rawValues, metric):
    rawValues = list(map(float, rawValues))
    normalisedValues = []
    # Normalise the raw data
    for x in rawValues:
        if (max(rawValues) - min(rawValues)) == 0:
            normX = 1
        else:
            if metric in ['HJ','MG']:
                normX = (x - min(rawValues)) / (max(rawValues) - min(rawValues))
            else:
                normX = 1 - ((x - min(rawValues)) / (max(rawValues) - min(rawValues)))
        normalisedValues.append(normX)

    return normalisedValues
def plot_comparison_graphs(tab, data,normalise):
    BDColor = ['#59DC23', ] * len(data['Dist. Bias'].tolist())
    HJColor = ['#2359DC', ] * len(data['Hallucination'].tolist())
    MGColor = ['#DC2359', ] * len(data['Gen. Miss Rate'].tolist())
    if not normalise:
        BDData = data['Dist. Bias']
        HJData = data['Hallucination']
        MGData = data['Gen. Miss Rate']
    else:
        data['Dist. Bias'] = normalise_data(data['Dist. Bias'], 'BD')
        data['Hallucination'] = normalise_data(data['Hallucination'], 'HJ')
        data['Gen. Miss Rate'] = normalise_data(data['Gen. Miss Rate'], 'MG')
    with tab:
        st.write("Selected evaluations for comparison:")
        st.write(data)

        BDFig = px.bar(x=data['Eval. ID'], y=data['Dist. Bias'],color_discrete_sequence=BDColor).update_layout(
                       xaxis_title=r'Evaluation ID', yaxis_title=r'Distribution Bias', title=r'Distribution Bias Comparison')
        st.plotly_chart(BDFig, theme="streamlit",use_container_width=True)

        HJFig = px.bar(x=data['Eval. ID'], y=data['Hallucination'],color_discrete_sequence=HJColor).update_layout(
                       xaxis_title=r'Evaluation ID', yaxis_title=r'Jaccard Hallucination', title=r'Jaccard Hallucination Comparison')
        st.plotly_chart(HJFig, theme="streamlit",use_container_width=True)

        MGFig = px.bar(x=data['Eval. ID'], y=data['Gen. Miss Rate'],color_discrete_sequence=MGColor).update_layout(
                       xaxis_title=r'Evaluation ID', yaxis_title=r'Generative Miss Rate', title=r'Generative Miss Rate Comparison')
        st.plotly_chart(MGFig, theme="streamlit",use_container_width=True)
        if normalise:

            Full3DFig = px.scatter_3d(data, x='Dist. Bias', y='Hallucination', z='Gen. Miss Rate',
                                      width=800, height=800,color='Eval. ID',title='3D Text-to-Image Model Bias Comparison')
            st.plotly_chart(Full3DFig, theme="streamlit",use_container_width=True)
