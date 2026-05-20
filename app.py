import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score # NUEVO: Importamos f1_score
import re

# --- Función para separar palabras pegadas ---
def limpiar_nombre(texto):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', texto).title()

# --- CONFIGURACIÓN DE PÁGINA Y CSS ---
st.set_page_config(page_title="Auditoría de Igualdad y Retención", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    div.stButton > button {
        background: linear-gradient(90deg, #0062ff 0%, #00d4ff 100%);
        color: white; border: none; padding: 10px 20px; border-radius: 8px;
        font-weight: bold; font-size: 16px; box-shadow: 0 4px 15px rgba(0, 98, 255, 0.3);
        transition: all 0.3s ease; width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 98, 255, 0.4);
        background: linear-gradient(90deg, #0052db 0%, #00c4ed 100%);
    }

    .stCheckbox [data-testid="stWidgetLabel"] p {
        font-size: 13.5px !important; line-height: 1.1; white-space: normal;
    }
    
    .modebar {display: none !important;}
    .gauge-card {
        background-color: #1e1e1e; border-radius: 25px; padding: 10px;
        border: 1px solid #333333; box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA DE LA APP ---
st.title("People Analytics: Auditoría de Equidad y Fuga")
st.markdown("Sube la base de datos de recursos humanos de tu organización para identificar riesgos de fuga y auditar posibles sesgos estructurales.")

# --- 1. CARGA UNIVERSAL DE ARCHIVOS ---
archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido, sep=None, engine='python')
    
    # Limpieza inicial de columnas fantasma de Excel
    df = df.dropna(how='all', axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # --- 2. PANEL DE MAPEO (DATA MAPPING) ---
    st.sidebar.header("1. Mapeo de Variables")
    st.sidebar.markdown("Indica cómo se llaman las columnas clave en tu archivo:")
    
    col_target = st.sidebar.selectbox("Columna de Baja/Fuga (Target):", df.columns)
    valores_target = df[col_target].unique()
    valor_fuga = st.sidebar.selectbox(f"¿Qué valor en '{col_target}' indica abandono?", valores_target)
    
    opciones_genero = ["Ninguna / No disponible"] + list(df.columns)
    col_genero = st.sidebar.selectbox("Columna de Género (Obligatoria para auditoría):", opciones_genero)
    
    df = df.dropna(subset=[col_target])
    df['Fuga'] = df[col_target].apply(lambda x: 1 if x == valor_fuga else 0)
    
    # --- FILTRO INTELIGENTE DE COLUMNAS ---
    cols_candidatas = [c for c in df.columns if c not in [col_target, 'Fuga']]
    cols_disponibles = []
    
    for c in cols_candidatas:
        if 'fecha' in c.lower() or 'date' in c.lower():
            continue
            
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            if df[c].nunique() > (len(df) * 0.85):
                continue
        
        cols_disponibles.append(c)
    
    # Imputación estrictamente matemática
    for col_name in cols_disponibles:
        if pd.api.types.is_numeric_dtype(df[col_name]):
            if df[col_name].min() == 0:
                df[col_name] = df[col_name].fillna(0)
            else:
                df[col_name] = df[col_name].fillna(df[col_name].median())
        else:
            df[col_name] = df[col_name].fillna("Desconocido")

    # --- 3. SELECCIÓN DE VARIABLES ---
    st.sidebar.write("---")
    st.sidebar.header("2. Configurar Modelo")
    
    if st.sidebar.button("Auto-Selección Inteligente", use_container_width=True):
        if len(cols_disponibles) > 0:
            with st.spinner('Analizando relevancia de todas las variables...'):
                X_all = df[cols_disponibles]
                y_all = df['Fuga']
                
                X_all_encoded = pd.get_dummies(X_all, drop_first=True)
                
                if not X_all_encoded.empty and X_all_encoded.shape[1] > 0:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
                    rf.fit(X_all_encoded, y_all)
                    
                    importances = rf.feature_importances_
                    col_importance = {col: 0.0 for col in cols_disponibles}
                    for feat, imp in zip(X_all_encoded.columns, importances):
                        for orig_col in cols_disponibles:
                            if feat == orig_col or feat.startswith(orig_col + '_'):
                                col_importance[orig_col] += imp
                                break
                    
                    top_10_size = min(10, len(cols_disponibles))
                    top_10_cols = sorted(col_importance, key=col_importance.get, reverse=True)[:top_10_size]
                    
                    if col_genero != "Ninguna / No disponible" and col_genero in cols_disponibles and col_genero not in top_10_cols:
                        if len(top_10_cols) == top_10_size:
                            top_10_cols[-1] = col_genero
                        else:
                            top_10_cols.append(col_genero)
                        
                    st.session_state['selected_vars'] = top_10_cols
                    
                    for col_name in cols_disponibles:
                        st.session_state[f"check_{col_name}"] = (col_name in top_10_cols)
                        
                    st.rerun()
                else:
                    st.sidebar.error("Los datos procesados no tienen un formato numérico compatible.")
        else:
            st.sidebar.error("No se han encontrado variables aptas para análisis predictivo.")

    if 'selected_vars' not in st.session_state:
        st.session_state['selected_vars'] = cols_disponibles[:min(5, len(cols_disponibles))]

    current_selection = []
    c1, c2 = st.sidebar.columns(2)
    
    for i, col_name in enumerate(cols_disponibles):
        col = c1 if i % 2 == 0 else c2
        es_genero = (col_name == col_genero)
        nombre_limpio = limpiar_nombre(col_name)
        valor_casilla = True if es_genero else (col_name in st.session_state['selected_vars'])
        
        if col.checkbox(nombre_limpio, value=valor_casilla, disabled=es_genero, key=f"check_{col_name}"):
            current_selection.append(col_name)
        elif es_genero:
            current_selection.append(col_name)
            
    st.session_state['selected_vars'] = current_selection

    # --- 4. ENTRENAMIENTO Y PRECISIÓN ---
    if len(current_selection) > 0 and not df.empty:
        X = df[current_selection]
        y = df['Fuga']
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        if not X_encoded.empty and X_encoded.shape[1] > 0:
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_encoded, y)
            preds = model.predict(X_encoded)
            
            # --- NUEVO: Cálculo de métricas ---
            acc = accuracy_score(y, preds) * 100
            f1 = f1_score(y, preds, zero_division=0) * 100
            
            st.sidebar.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <p style="margin-bottom: 0px; font-size: 16px;">Precisión Global:</p>
                    <h1 style="color: #0062ff; margin-top: 0px; font-size: 40px;">{acc:.1f}%</h1>
                    <p style="margin-bottom: 5px; font-size: 16px; color: #27ae60;"><b>F1-Score: {f1:.1f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Nota explicativa sobre el F1-Score para RR.HH.
            st.sidebar.caption("💡 *El **F1-Score** es la métrica clave de fiabilidad: mide la capacidad del modelo para detectar fugas reales de talento sin generar falsas alarmas.*")
            
            if len(current_selection) > 15:
                st.sidebar.warning("⚠️ Modelo muy complejo: Riesgo de sobreajuste (ruido estadístico).")
            elif len(current_selection) < 3:
                st.sidebar.warning("⚠️ Modelo demasiado simple: Riesgo de sesgo por falta de información.")
            else:
                st.sidebar.success("✅ Modelo equilibrado y óptimo.")
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Top Predictoras:")
            coef_df = pd.DataFrame({'var': X_encoded.columns, 'abs_coef': np.abs(model.coef_[0])})
            top_3 = coef_df.sort_values(by='abs_coef', ascending=False).head(min(3, len(coef_df)))
            for i, row in enumerate(top_3['var'], 1):
                clean_name = limpiar_nombre(row.split('_')[0])
                st.sidebar.write(f"**{i}º** {clean_name}")
        else:
            st.sidebar.error("Las variables seleccionadas no aportan información válida.")
            acc = 0
    else:
        st.sidebar.warning("Selecciona al menos una variable para entrenar el modelo.")
        acc = 0

    st.sidebar.markdown("---")
    st.sidebar.caption("🛡️ **Cumplimiento RGPD:** Esta herramienta procesa los datos temporalmente en memoria local. No se almacena ni comparte ninguna información, garantizando la privacidad y minimización de datos corporativos.")

    # --- 5. PANEL PRINCIPAL (SIMULADOR DE GEMELOS DIGITALES) ---
    if acc > 0:
        col_main1, col_main2 = st.columns([1, 1])

        with col_main1:
            st.subheader("Simulador de Perfiles")
            inputs_usuario = {}
            
            variables_ordenadas = []
            if col_genero != "Ninguna / No disponible" and col_genero in current_selection:
                variables_ordenadas.append(col_genero)
            
            for v in current_selection:
                if v != col_genero:
                    variables_ordenadas.append(v)
            
            for var in variables_ordenadas:
                nombre_limpio_slider = limpiar_nombre(var)
                
                if df[var].dtype == 'object' or pd.api.types.is_string_dtype(df[var]) or df[var].nunique() < 5:
                    opciones = sorted(df[var].astype(str).unique().tolist())
                    inputs_usuario[var] = st.selectbox(f"{nombre_limpio_slider}:", opciones, key=f"input_{var}")
                else:
                    min_v = df[var].min()
                    max_v = df[var].max()
                    avg_v = df[var].mean()
                    
                    if np.issubdtype(df[var].dtype, np.integer):
                        inputs_usuario[var] = st.slider(f"{nombre_limpio_slider}:", int(min_v), int(max_v), int(avg_v), step=1, key=f"input_{var}")
                    else:
                        inputs_usuario[var] = st.slider(f"{nombre_limpio_slider}:", float(min_v), float(max_v), float(avg_v), key=f"input_{var}")

            user_df = pd.DataFrame([inputs_usuario])
            user_encoded = pd.get_dummies(user_df).reindex(columns=X_encoded.columns, fill_value=0)
            prob = model.predict_proba(user_encoded)[0][1] * 100

        with col_main2:
            st.subheader("Riesgo de Fuga Calculado")
            color_arco = "#27ae60" if prob < 30 else "#f1c40f" if prob < 70 else "#e74c3c"

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                number = {'suffix': "%", 'font': {'size': 80, 'color': "white", 'family': "'Outfit', sans-serif", 'weight': 700}},
                gauge = {
                    'axis': {'range': [None, 100], 'visible': False},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'steps': [{'range': [0, prob], 'color': color_arco}],
                    'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': prob}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=30, r=30, t=30, b=0), height=350)

            st.markdown('<div class="gauge-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            if prob < 30:
                st.success("🟢 **Riesgo Bajo:** Condiciones de retención óptimas para este perfil. No se requiere acción inmediata.")
            elif prob < 70:
                st.warning("🟡 **Riesgo Moderado:** Se recomienda monitorizar la situación y evaluar planes de retención.")
            else:
                st.error("🔴 **Riesgo Crítico:** Alta probabilidad de fuga. Intervención inmediata sugerida (revisar equidad y desarrollo).")

else:
    st.info("Por favor, arrastra y suelta tu archivo CSV en la caja superior para comenzar.")
