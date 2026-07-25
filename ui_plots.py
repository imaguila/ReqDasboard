import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def render_scatter_plot(df, x, y, size, color_col, show_ids, key):
    import numpy as np
    import pandas as pd
    import plotly.express as px

    df = df.copy()

    # -----------------------------
    # Hover dinámico
    # -----------------------------
    hover_data = {}
    if "score" in df.columns:
        hover_data["score"] = ':.3f'
    if "score_topsis" in df.columns:
        hover_data["score_topsis"] = ':.3f'
    if "count" in df.columns:
        hover_data["count"] = True
    if "cluster" in df.columns:
        hover_data["cluster"] = True

    # -----------------------------
    # Detectar discreto / continuo
    # -----------------------------
    is_discrete = False

    if color_col and color_col in df.columns:
        if color_col == "group_label":
            is_discrete = True
            df[color_col] = df[color_col].astype(str)

        elif pd.api.types.is_object_dtype(df[color_col]):
            is_discrete = True
            df[color_col] = df[color_col].astype(str)

    # ======================================================
    # ✅ CASO 1: CONTINUO (Weighted, TOPSIS, etc.)
    # ======================================================
    if not is_discrete:

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=color_col,
            color_continuous_scale=px.colors.sequential.Viridis,
            text="label" if show_ids else None,
            hover_data=hover_data
        )

        # ✅ aplicar opacidad si hay selección
        if "highlight" in df.columns and df["highlight"].any():
            opacity_vals = np.where(df["highlight"], 1.0, 0.25)
            fig.update_traces(marker=dict(opacity=opacity_vals))

        fig.update_layout(
            legend_title_text=color_col if color_col else ""
        )

    # ======================================================
    # ✅ CASO 2: DISCRETO (Clustering, Ranking)
    # ======================================================
    else:




        unique_vals = sorted(df[color_col].dropna().unique().tolist())
        palette = px.colors.qualitative.Plotly
        color_map = {}

        palette_idx = 0
        for v in unique_vals:
            if str(v).startswith("No match"):
                color_map[v] = "#b0b0b0"   # gris para las no coincidencias
            else:
                color_map[v] = palette[palette_idx % len(palette)]
                palette_idx += 1

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=color_col,
            text="label" if show_ids else None,
            hover_data=hover_data,
            color_discrete_map=color_map
        )

        # ✅ aplicar opacidad también aquí
        if "highlight" in df.columns and df["highlight"].any():
            opacity_vals = np.where(df["highlight"], 1.0, 0.25)
            fig.update_traces(marker=dict(opacity=opacity_vals))

        fig.update_layout(legend_title_text="Groups")

    # -----------------------------
    # Estética
    # -----------------------------
    fig.update_traces(
        textposition="top right",
        textfont=dict(size=10),
        marker=dict(size=10),
        mode='markers+text' if show_ids else 'markers'
    )

    st.plotly_chart(fig, use_container_width=True, key=key)

def plot_radar(selected_df, available_metrics, group_col=None):
    st.markdown("---")

    df_for_compare = selected_df

    
    # -------------------------------
    # Selección de IDs SOLO dentro del grupo elegido
    # -------------------------------
    opciones_id = df_for_compare["id"].unique()
    compare_ids = st.multiselect("👆 Pick solutions to compare", opciones_id)

    if len(compare_ids) < 2:
        st.info("Select at least 2 solutions to compare")
        return


    compare_df = df_for_compare[df_for_compare["id"].isin(compare_ids)].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Comparative Profile",
        "👥 Stakeholder Impact",
        "📋 Requirement Composition"#,
       # "🤝 Stakeholder–Requirement Alignment"
    ])

    # ---------- PERFORMANCE ----------
    with tab1:
        st.subheader("Custom Trade-off Comparison")
        
        numeric_cols = [m for m in available_metrics if pd.api.types.is_numeric_dtype(selected_df[m])]
        selected_radar_metrics = st.multiselect(
            "Select metrics (at least 3)", 
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else None,
            key="perf_metrics"
        )

        if len(selected_radar_metrics) >= 3:
            metric_goals = {}
            cols = st.columns(len(selected_radar_metrics))
            
            for idx, m in enumerate(selected_radar_metrics):
                with cols[idx]:
                    goal = st.selectbox(f"Goal {m}", ["Maximize", "Minimize"], key=f"radar_g_{m}")
                    metric_goals[m] = goal

            radar_df = compare_df.copy()
            low, high = 0.1, 0.9

            for m in selected_radar_metrics:
                mi, ma = radar_df[m].min(), radar_df[m].max()
                if ma > mi:
                    norm = (radar_df[m] - mi) / (ma - mi)
                    if metric_goals[m] == "Minimize":
                        norm = 1.0 - norm
                    radar_df[m] = low + (norm * (high - low))
                else:
                    radar_df[m] = 0.5

            fig_perf = go.Figure()
            for _, row in radar_df.iterrows():
                val = row[selected_radar_metrics].tolist()
                val.append(val[0])
                fig_perf.add_trace(go.Scatterpolar(
                    r=val,
                    theta=selected_radar_metrics + [selected_radar_metrics[0]],
                    fill=None,
                    mode='lines+markers',
                    name=f"ID {int(row['id'])}"
                ))

            fig_perf.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        else:
            st.warning("Select at least 3 metrics.")

 # ---------- STAKEHOLDER ----------
    with tab2:
        st.subheader("Coverage per Stakeholder")
        
        stcov_cols = [c for c in selected_df.columns if c.startswith("stcov_")]

        # ----------------------------------
        # ✅ No hay stakeholders
        # ----------------------------------
        if not stcov_cols:
            st.info("No stakeholder coverage columns (stcov_...) found in dataset.")

        else:

            # ----------------------------------
            # ✅ ordenar por relevancia
            # ----------------------------------
            stcov_cols = sorted(
                stcov_cols,
                key=lambda c: selected_df[c].mean(),
                reverse=True
            )

            # ----------------------------------
            # ✅ selección manual (clave)
            # ----------------------------------
            selected_st = st.multiselect(
                "👆 Select stakeholders to display",
                stcov_cols,
                default=stcov_cols[:min(6, len(stcov_cols))]
            )

            # ✅ aplicar selección
            if selected_st:
                stcov_cols = selected_st
            else:
                stcov_cols = []

            st.caption(f"Showing {len(stcov_cols)} stakeholders")

            # ----------------------------------
            # ⚠️ asegurar mínimo
            # ----------------------------------
            if len(stcov_cols) < 3:
                st.warning("Need at least 3 stakeholders to create a radar chart.")
            else:
                cov_df = compare_df.copy()
                low, high = 0.1, 0.9

                for c in stcov_cols:
                    mi, ma = cov_df[c].min(), cov_df[c].max()
                    if ma > mi:
                        norm = (cov_df[c] - mi) / (ma - mi)
                        cov_df[c] = low + (norm * (high - low))
                    else:
                        cov_df[c] = 0.5

                fig_cov = go.Figure()

                for _, row in cov_df.iterrows():
                    val = row[stcov_cols].tolist()
                    val.append(val[0])

                    fig_cov.add_trace(go.Scatterpolar(
                        r=val,
                        theta=stcov_cols + [stcov_cols[0]],
                        fill=None,
                        mode='lines+markers',
                        name=f"ID {int(row['id'])}"
                    ))

                fig_cov.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True
                )

                st.plotly_chart(fig_cov, use_container_width=True)
# ---------- NUEVA PESTAÑA: REQUISITOS ----------
    with tab3:
        st.subheader("Requirements Included in Selected Solutions")
        
        # Detectamos dinámicamente cualquier columna que empiece por "req_"
        req_cols = [c for c in selected_df.columns if c.startswith("req_")]
        
        if not req_cols:
            st.info("No requirement columns (req_...) found in dataset.")
        else:
            # Preparamos los datos: Filas como IDs de solución y columnas como Requisitos
            req_df = compare_df.set_index("id")[req_cols].copy()
            
            # Convertimos el ID a string para el eje Y
            req_df.index = [f"ID {int(i)}" for i in req_df.index]
            
            # MAPA DE COLOR CON CONTRASTE ALTO:
            fig_req = px.imshow(
                req_df,
                labels=dict(x="Requirements", y="Solutions", color="Status"),
                x=req_cols,
                y=req_df.index,
                color_continuous_scale=[[0, "#e0e0e0"], [1, "#00e676"]] # Gris Claro vs Verde Brillante Puro
            )
            
            # Configuramos el diseño del gráfico de forma correcta
            fig_req.update_layout(
                template="plotly_white", # Fondo blanco limpio
                coloraxis_showscale=False, # Ocultamos la barra lateral de colores
                xaxis=dict(
                    tickangle=-45, 
                    showgrid=False, # Quitamos líneas de cuadrícula del fondo
                    tickfont=dict(size=11, color="black") # ¡CORREGIDO AQUÍ! (Era tickfont, no textfont)
                ),
                yaxis=dict(
                    autorange="reversed", 
                    showgrid=False,
                    tickfont=dict(size=11, color="black") # ¡CORREGIDO AQUÍ! (Era tickfont, no textfont)
                ),
                margin=dict(l=50, r=50, t=30, b=50) # Ajustamos márgenes
            )
            
            # Añadimos bordes blancos muy marcados para separar las celdas claramente
            fig_req.update_traces(
                xgap=3, 
                ygap=3,
                hovertemplate="<b>%{y}</b><br>Requirement: %{x}<br>Status: %{z} (1=Included, 0=Excluded)<extra></extra>"
            )
            
            st.plotly_chart(fig_req, use_container_width=True)


# ---------- NUEVA PESTAÑA: ALINEACIÓN STAKEHOLDERS + FILA RESUMEN ----------
#    with tab4:
#        st.subheader("Stakeholder-Requirement Alignment Matrix")
#        st.write("Visualizing which requirements satisfy each stakeholder's specific interests and their final release status.")

        # 1. Identificar columnas
#        req_cols = [c for c in selected_df.columns if c.startswith("req_")]
#        st_cols = [c for c in selected_df.columns if c.startswith("stcov_")]

        # Desplegable para analizar una solución concreta
#        focus_id = st.selectbox("Select Solution to analyze alignment", compare_df["id"].unique(), key="align_sel")
#        row_focus = compare_df[compare_df["id"] == focus_id].iloc[0]

        # Matriz real (requisitos x stakeholders) calculada a partir del
        # dataset del problema (vij) en problem.py::calcular_matriz_solicitud,
        # y guardada en session_state por input_panel.py. Sustituye al
        # antiguo mapeo simulado basado en hash(st_name + req_name).
#        matriz_solicitud = st.session_state.get("matriz_solicitud")

#        if not req_cols or not st_cols:
#            st.info("Required data columns (req_ or stcov_) missing.")
#        elif matriz_solicitud is None:
#            st.info(
#                "No stakeholder-request data available for this dataset "
#                "(e.g. no problem file was loaded, or you are using an "
#                "uploaded CSV). Alignment cannot be computed."
#            )
#        else:

#            alignment_data = []
#            # Generamos las filas de los Stakeholders normales
#            for st_name in st_cols:
#                # "stcov_cv1" -> "cv1", para indexar en matriz_solicitud
#                cliente = st_name.replace("stcov_", "")

#                if cliente not in matriz_solicitud.columns:
                    # Por seguridad, si no encontramos el stakeholder en la
                    # matriz real, lo tratamos como "no solicitado" en vez
                    # de simularlo.
#                    row_values = [0] * len(req_cols)
#                    alignment_data.append(row_values)
#                    continue

#                row_values = []
#                for j, req_name in enumerate(req_cols):
#                    proposed_by_stake = bool(matriz_solicitud.iloc[j][cliente])
#                    is_included = row_focus[req_name] == 1

#                    if proposed_by_stake and is_included:
#                        val = 2  # Solicitado e incluido (Verde brillante)
#                    elif proposed_by_stake and not is_included:
#                        val = 1  # Solicitado pero fuera (Gris medio)
#                    else:
#                        val = 0  # No solicitado (Gris muy claro)
#                    row_values.append(val)
#                alignment_data.append(row_values)

            # --- LA FILA RESUMEN ---
            # Añadimos la fila final que mira directamente si el req está en la solución (1) o no (0)
#            summary_row = []
#            for req_name in req_cols:
#                if row_focus[req_name] == 1:
#                    summary_row.append(3) # Incluido en el release (Verde Oscuro)
#                else:
#                    summary_row.append(1) # Fuera del release (Gris Medio)
#            alignment_data.append(summary_row)

            # Creamos la lista de nombres para el eje Y incluyendo nuestro resumen
#            y_labels = [s.replace("stcov_", "Stakeholder ") for s in st_cols] + ["📦 RELEASE STATUS"]

            # Crear DataFrame para el Heatmap
#            align_df = pd.DataFrame(alignment_data, index=y_labels, columns=req_cols)

            # 2. Construir el Heatmap con Escala de 4 colores discretos
#            fig_align = px.imshow(
#                align_df,
#                labels=dict(x="Requirements", y="Alignment Status", color="Status"),
#                x=req_cols,
#                y=y_labels,
                # Definimos los cortes exactos de color para 0, 1, 2 y 3
#                color_continuous_scale=[
#                    [0.0, "#f8f9fa"],   # 0: Sin interés (Blanco/Gris suave)
#                    [0.33, "#adb5bd"],  # 1: Fuera del Release / Deuda (Gris medio)
#                    [0.66, "#00e676"],  # 2: Solicitado e Incluido (Verde brillante)
#                    [1.0, "#00695c"]    # 3: Fila Resumen - ¡En el Release! (Verde Oscuro Azulado)
#                ]
#            )

            # Diseño limpio del gráfico
#            fig_align.update_layout(
#               template="plotly_white",
#                coloraxis_showscale=False, # Ocultamos barra de escala continua
#                xaxis=dict(tickangle=-45, tickfont=dict(size=11, color="black")),
#                yaxis=dict(tickfont=dict(size=11, color="black")),
#                height=450
#            )

            # Espaciado de celdas y caja de información al pasar el ratón (Hover)
#            fig_align.update_traces(
#                xgap=3, ygap=3,
#                hovertemplate="<b>%{y}</b><br>Requirement: %{x}<extra></extra>"
#            )

#            st.plotly_chart(fig_align, use_container_width=True)
            
            # Leyenda explicativa interactiva en columnas abajo del gráfico
#            c1, c2, c3= st.columns(3)
#            c1.markdown("⚪ **Not requested**")
#            c2.markdown("🔘 **Requested (Not included) / Excluded from the Release**")
#            c3.markdown("🟢 **Requested and Included (Stakeholder)**")
          #  c4.markdown("🌲 **Included in the Final Release (Summary Row)**")

# --------------------------------------------