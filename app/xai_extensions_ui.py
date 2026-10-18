"""Bloc complémentaire pour Streamlit : permutation, gain, LIME et comparaison."""
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

def render_extra_xai(api_call):
    st.markdown('### Comparaison des méthodes d’importance')
    data,err=api_call('GET','/explain/importances')
    if err:
        st.warning(f'Importances non disponibles : {err}')
    else:
        df=pd.DataFrame(data)
        columns={'importance_xgboost':'Importance propre XGBoost',
                 'permutation_xgboost':'Permutation XGBoost (chute AUC)',
                 'permutation_logreg':'Permutation régression logistique (chute AUC)'}
        choice=st.selectbox('Méthode',list(columns),format_func=lambda c:columns[c],key='xai_importance_method')
        top=df.nlargest(12,choice).sort_values(choice)
        fig=px.bar(top,x=choice,y='variable',orientation='h',
                title=columns[choice],labels={choice:'Importance','variable':''})
        fig.update_layout(height=520)
        st.plotly_chart(fig,use_container_width=True)
        st.caption('La permutation porte sur les 36 variables originales. Les importances propres XGBoost regroupent les colonnes encodées.')
        shap_path=__import__('pathlib').Path(__file__).resolve().parents[1]/'artifacts/shap_global_importance.csv'
        if shap_path.exists():
            imp=pd.read_csv(shap_path).set_index('variable')['importance_shap']
            ranks=pd.DataFrame({'SHAP XGBoost':imp.rank(ascending=False)})
            for c,label in columns.items():
                ranks[label]=df.set_index('variable')[c].rank(ascending=False)
            st.dataframe(ranks.sort_values('SHAP XGBoost').head(12).round().astype(int),use_container_width=True)

    st.divider()
    st.markdown('### LIME et comparaison avec SHAP')
    idx=st.number_input('Étudiant à expliquer avec LIME',min_value=0,max_value=884,value=0,step=1,key='xai_lime_student')
    if st.button('Comparer LIME et SHAP',key='xai_compare_btn'):
        with st.spinner('Calcul LIME...'):
            lime,err=api_call('GET',f'/explain/lime/{int(idx)}')
        if err:
            st.error('LIME indisponible : '+str(err));return
        student,error=api_call('GET',f'/students/{int(idx)}')
        if error:st.error(str(error));return
        shap_result,error=api_call('POST','/explain/shap',json={'modele':'xgboost','seuil':.3,'etudiant':student['etudiant']})
        if error:st.error(str(error));return
        st.metric('Risque XGBoost',f"{lime['probabilite_decrochage']:.2%}")
        conditions=pd.DataFrame(lime['conditions'])
        fig=px.bar(conditions.iloc[::-1],x='poids',y='condition',orientation='h',
                   color='poids',color_continuous_scale='RdBu_r',
                   title='Explication LIME (poids locaux)')
        fig.update_layout(height=540,coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
        sv=pd.Series(shap_result['contributions'],dtype=float)
        lv=pd.Series(lime['contributions'],dtype=float).reindex(sv.index).fillna(0)
        selected=list(dict.fromkeys(list(sv.abs().nlargest(5).index)+list(lv.abs().nlargest(5).index)))
        compare=pd.DataFrame({'SHAP (log-odds)':sv[selected], 'LIME (probabilite)':lv[selected]})
        compare['Même sens']=np.where(np.sign(compare.iloc[:,0])==np.sign(compare.iloc[:,1]),'Oui','Non')
        st.dataframe(compare.round(3),use_container_width=True)
        st.caption('Comparer les signes, pas les amplitudes : SHAP utilise les log-odds et LIME une approximation locale de la probabilité.')
        plus=sv[sv>0].sort_values(ascending=False)
        minus=sv[sv<0].sort_values()
        parts=[]
        if not plus.empty: parts.append('le risque augmente surtout avec '+plus.index[0])
        if not minus.empty: parts.append('il diminue surtout avec '+minus.index[0])
        st.info('Synthèse automatique : '+ '; '.join(parts)+'. Ces effets ne prouvent pas une causalité.')
