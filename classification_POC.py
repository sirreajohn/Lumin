# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:03:14 2021

@author: mahesh
"""

#------------------basic_Libs-----------------
import pandas as pd
import random

#-----------------------preprocessing-----------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#--------------------------algorithms-----------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#------------------------------metrics------------------------
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_curve,auc
from tqdm import tqdm

#---------------------STREAMLIT AND PLOTLY-------------------------
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff 

#-----------------------------Start here-------------------------------
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Project Lumin -- Classification report")
st.markdown("this is a automated report generated by Project Lumin, an Unified Machine Learning Framework.")
#------------User defined-------------------
data_path = st.file_uploader("drop the data set here: ", type = ["csv"])
if data_path:
    data = pd.read_csv(data_path)
    tar_var = st.selectbox("choose Y var(prediction)",list(data.columns))
    if st.button("press here"):
        target = data[tar_var]
        
        #--------------------------------------preprocessing-------------------------------
        drop_str = [col for col in data.columns if type(data[col][0]) == str]
        data_head = data.copy(deep = True)      #is used in presentation
        data = data.drop(drop_str,axis = 1)
        data = data.drop(target.name, axis = 1) #dropping y from data
        corr_mat = data_head.corr()             # for later use in presentation
        
        
        pca = False                             # FIX THIS!
        if len(data.columns) > 2:
            pca = True
        
        
        
        #------------------------------scaling------------------------
        sc_1 = StandardScaler()
        x_scaled = sc_1.fit_transform(data)
        
        #----------------------------splits---------------------
        x_train,x_test,y_train,y_test = train_test_split(x_scaled,target ,test_size = 0.2, random_state = 177013)
        
        #-----------------------PCA only if >2 cols---------------
        if pca == True:
            pca = PCA(n_components = 2)
            x_train = pd.DataFrame(data = pca.fit_transform(x_train),columns = ['pc1',"pc2"]).iloc[:,:].values
            x_test = pca.transform(x_test)
        
        #----------------------------algorithms-----------------------------NB is disqualified 
        # (has some reservations about neg values)
        
        classification_models = {"LR":LogisticRegression(),"SVC":SVC(kernel = "rbf"),
                                 "DTC":DecisionTreeClassifier(),
                                 "RFC":RandomForestClassifier(n_estimators = 500),
                                 "XGBC":XGBClassifier(n_estimators = 500)}
        
        metric_dict = {}
        accu_dict = {}
        for name,algorithm in tqdm(classification_models.items()):
            model = algorithm
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            metric_dict[name] = {"precision":round(precision_score(y_test,y_pred,pos_label=y_pred[0]),2),
                                 "recall":round(recall_score(y_test,y_pred,pos_label=y_pred[0]),2),
                                 "f1_score":round(f1_score(y_test,y_pred,average = 'micro'),2),
                                 "accuracy":accuracy_score(y_test,y_pred),
                                 "confusion":confusion_matrix(y_test,y_pred),
                                 "ROC_Vals" :roc_curve(y_test,y_pred,pos_label=y_pred[0])}
            accu_dict[name] = accuracy_score(y_test,y_pred)
            
        #-------------------------helper FUNCTIONS---------------------
        def list_maker(metric_dict,keyword = "accuracy"):
            key_list = list(metric_dict.keys())
            return [metric_dict[key][keyword] for key in key_list]
        
        def random_color(metric_dict):
            return ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(metric_dict))]
            
        
            
        
        metric_df = pd.DataFrame(metric_dict).drop(["confusion","ROC_Vals"],axis = 0)
        metric_df.reset_index(inplace = True)
        
        
        #--------------------------------------------- presentation and graphs -----------------------------------0
        
        #-------------------------------------view data --------------------------
        
        st.header("Lets look at what we are dealing with ")
        st.dataframe(data_head.head())
        
        #-----------------------------corelation_plot----------------------------------
        
        st.header("Corelation Plot")
        st.markdown("zoom if intelligible")
        corr_val = data.corr()
        corr = ff.create_annotated_heatmap(y = corr_val.index.tolist(),
                                           x = corr_val.columns.tolist(),z = corr_val.values)
        for i in range(len(corr.layout.annotations)):
            corr.layout.annotations[i].font.size = 8
            corr.layout.annotations[i].text = str(round(float(corr.layout.annotations[i].text),4))
        corr.update_layout(width = 800,height = 800)
        st.plotly_chart(corr)
        st.header("METRICS FOR CLASSIFICATION ALGORITHMS")
        
        #------------------------metric_table-----------------
        table = ff.create_table(metric_df)
        table.update_layout(width = 1350)
        st.plotly_chart(table)
        
        
        #--------------heatmaps------------------------------
        st.markdown("### CONFUSION MATRICES")
        
        fig = make_subplots(rows = 1,cols =len(metric_df.columns[1:].values),
                            shared_yaxes=True,horizontal_spacing=0.05,
                            subplot_titles= metric_df.columns[1:].values)
        
        annot_var = []
        axis_count = 0
        row_col = []
        for row in range(1,2):
            for col in range(1,6):
                row_col.append([row,col])
        row_col_pos = 0    
        for al in metric_df.columns[1:].values:
            heatmap2 = ff.create_annotated_heatmap(z = metric_dict[al]["confusion"],x = ["1_pred","0_pred"],
                                               y = ["1_true","0_true"],
                                               annotation_text= metric_dict[al]["confusion"])
            fig.add_trace(heatmap2.data[0],row_col[row_col_pos][0],row_col[row_col_pos][1])
            annot_temp = list(heatmap2.layout.annotations)
            axis_count = axis_count + 1
            row_col_pos = row_col_pos+1
            for k  in range(len(annot_temp)):
                annot_temp[k]['xref'] = "x"+str(axis_count)
                annot_temp[k]['yref'] = 'y'+str(axis_count)
            annot_var= annot_var + annot_temp
                
        lo = list(fig['layout']["annotations"]) + annot_var
        fig.update_layout(annotations = lo,autosize = True,width = 1350)
        
        st.plotly_chart(fig)
        
        
        #------------scatter plots----------------
        
        
        
        fpr, tpr,thres = roc_curve(y_test,y_pred,pos_label=y_pred[0])
        
        scatter_plot = go.Figure(go.Scatter(x = [0,1], y = [0,1], mode = "lines", name = "ref"))
        
        for al in metric_df.columns[1:].values:
            
            AUC_val = auc(metric_dict[al]["ROC_Vals"][0].tolist(),metric_dict[al]["ROC_Vals"][1].tolist())
                
            scat = go.Scatter(x = metric_dict[al]["ROC_Vals"][0].tolist(),
                              y = metric_dict[al]["ROC_Vals"][1].tolist(),
                              name = f"{al} - AUC val - {AUC_val:.2f}")
            scatter_plot.add_trace(scat)
            
        scatter_plot.update_layout(width = 1300,height = 500)
        st.header("ROC_curves")
        st.plotly_chart(scatter_plot)
        
        
        #-------------funnel-chart-----------------   
        
        st.header("Recommendations")
        st.markdown("the percent below classifier represents recommended probability for classifier")
        accu_dict = dict(sorted(accu_dict.items(), key=lambda item: item[1],reverse=True))
        funnel = go.Figure(go.Funnelarea(values  = list(accu_dict.values()),text = list(accu_dict.keys())))
        funnel.update_layout(showlegend = False)
        st.plotly_chart(funnel)























