import streamlit as st
import pandas as pd
#import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.cluster import HDBSCAN

from input_panel import render_input_panel
from metrics_catalog import get_metric_sets
from ui_plots import render_scatter_plot, plot_radar

# --------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------
st.set_page_config(layout="wide")
st.title("ANRP Decision Space Explorer")
DATA_PATH = "data"

# --------------------------------------------
# DATA SOURCE + DATA PREPARATION
# --------------------------------------------

df = render_input_panel()

# --------------------------------------------
# METRICS
# --------------------------------------------
available_opt, available_qual, available_metrics = get_metric_sets(df, DATA_PATH)

# --------------------------------------------
# SESSION STATE
# --------------------------------------------
if "groups" not in st.session_state:
    st.session_state.groups = []

if "saved_sois" not in st.session_state:
    st.session_state.saved_sois = []

if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = []

if "pending_clear_selected_ids" not in st.session_state:
    st.session_state.pending_clear_selected_ids = False

if "focus_mode" not in st.session_state:
    st.session_state.focus_mode = False


if "focus_locked" not in st.session_state:   # para highlight
    st.session_state.focus_locked = False


if "combine_sois_mode" not in st.session_state:
    st.session_state.combine_sois_mode = False

if "selected_sois_for_combination" not in st.session_state:
    st.session_state.selected_sois_for_combination = []


if "active_combined_soi" not in st.session_state:
    st.session_state.active_combined_soi = None

if "consensus_threshold" not in st.session_state:
    st.session_state.consensus_threshold = 0.5

if "pending_reset_active_soi" not in st.session_state:
    st.session_state.pending_reset_active_soi = False


# --------------------------------------------
# VISUAL WORKSPACE
# --------------------------------------------
used_now = [m for g in st.session_state.groups for m in g if m]
remaining = [m for m in available_metrics if m not in used_now]

st.sidebar.markdown("## 🗺️ Visual Workspace")
st.sidebar.caption("Manage 2D views of the decision space")

# Estado del workspace
n_maps = len(st.session_state.groups)
can_add_map = len(remaining) >= 2
can_reset_workspace = n_maps > 0

# Texto de ayuda contextual


st.sidebar.caption(
    f"Active maps: {n_maps} · Remaining metrics available for new maps: {len(remaining)}"
)


# Botones en paralelo de colores
st.html("""
    <style>
    /* Estilo para el botón de reset usando su clave única */
    div[data-testid="stActionButtonElement"] button[key="reset_workspace_btn"] {
        color: #ff4b4b;
        border-color: #ff4b4b;
    }
    </style>
""")


# col_reset, col_add = st.sidebar.columns(2)
col_reset, col_add = st.sidebar.columns([0.35, 0.65])
with col_reset:
    if st.button(
        "Reset maps",
        use_container_width=True,
        disabled=not can_reset_workspace,
        key="reset_workspace_btn",
        type="secondary"
    ):
        st.session_state.groups = []
        st.session_state.show_comparison = False
        st.rerun()

with col_add:
    if st.button(
        "New decision map",
        use_container_width=True,
        disabled=not can_add_map,
        key="add_map_btn",
        type="primary"
    ):
        st.session_state.groups.append([remaining[0], remaining[1], None])
        st.rerun()

if not can_add_map:
    st.sidebar.info("No remaining available metrics")

if can_reset_workspace:
    st.sidebar.caption("'Reset' clears maps, but keeps data")

show_ids = st.sidebar.checkbox(
    "Show IDs on plots",
    value=False,
    help="Display solution identifiers directly on the maps."
)


# --------------------------------------------
# CONTEXT FRAMING
# --------------------------------------------
st.sidebar.markdown("## 🎛️ Context Framing")

filtered_df = df.copy()

# -------- OPTIMIZATION METRICS --------
st.sidebar.markdown("#### 🔘 Optimization objectives")

for m in available_opt:
    if pd.api.types.is_numeric_dtype(df[m]):
        min_v, max_v = float(df[m].min()), float(df[m].max())

        if min_v != max_v:
            val_range = st.sidebar.slider(
                f"{m}",
                min_v,
                max_v,
                (min_v, max_v),
                key=f"opt_{m}"
            )

            filtered_df = filtered_df[
                (filtered_df[m] >= val_range[0]) &
                (filtered_df[m] <= val_range[1])
            ]

# -------- QUALITY METRICS --------
st.sidebar.markdown("#### 🔵 :blue[Quality Indicators]")

for m in available_qual:
    if pd.api.types.is_numeric_dtype(df[m]):
        min_v, max_v = float(df[m].min()), float(df[m].max())

        if min_v != max_v:
            val_range = st.sidebar.slider(
                f"{m}",
                min_v,
                max_v,
                (min_v, max_v),
                key=f"qual_{m}"
            )

            filtered_df = filtered_df[
                (filtered_df[m] >= val_range[0]) &
                (filtered_df[m] <= val_range[1])
            ]

# --------------------------------------------
# SELECCIÓN
# -------------------------------------------- 
st.sidebar.markdown("## 🔍 SOI Identification Lens")
mode_label = st.sidebar.selectbox(
    "🔍Analytical Lens",
    [
        " Exploratory view",
        "👁️ Preference lens (MCDA)",
        "✨ Diversity lens",
        "⚡ Efficiency lens",
        "💡 Domain-specific lens",
    ],
    label_visibility="collapsed"
)

mode_map = {
    " Exploratory view": "None",
    "👁️ Preference lens (MCDA)": "MCDM",
    "✨ Diversity lens": "Clustering",
    "⚡ Efficiency lens": "Efficiency-Ratio",
    "💡 Domain-specific lens": "Ranking-based",
}


mode = mode_map[mode_label]

lens_locked = st.session_state.get(
    "focus_locked",
    False
)

# selected_df es siempre el subconjunto sobre el que trabaja la lente activa.
# Se inicializa como copia EXACTA del framing (filtered_df): ninguna lente
# debe leer nunca de df ni saltarse filtered_df.
selected_df = filtered_df.copy()
threshold = 0

# IMPORTANTE: esta variable controla el color en los plots
color_col = None

# --------------------------------------------
# MCDM 
# --------------------------------------------
if mode == "MCDM":

    st.sidebar.markdown("### Multi-criteria Preference")

    # -------------------------------
    # Método de evaluación
    # -------------------------------
    method = st.sidebar.radio(
        "Method",
        ["Weighted Sum", "TOPSIS"],
        disabled=lens_locked
    )

    # -------------------------------
    # Selección de criterios
    # -------------------------------
    m_max = st.sidebar.multiselect(
        "Maximize",
        available_qual,
        disabled=lens_locked
    )
    m_min = st.sidebar.multiselect(
        "Minimize",
        [m for m in available_qual if m not in m_max],
        disabled=lens_locked
    )

    criteria = m_max + m_min

    if criteria:

        df_temp = selected_df.copy()

        # =====================================================
        # ✅ WEIGHTED SUM
        # =====================================================
        if method == "Weighted Sum":

            score = 0

            for m in criteria:
                mi, ma = df_temp[m].min(), df_temp[m].max()
                norm = (df_temp[m] - mi) / (ma - mi) if ma > mi else 0

                if m in m_max:
                    score += norm
                else:
                    score -= norm

            df_temp["score"] = score
            score_col = "score"

        # =====================================================
        # ✅ TOPSIS
        # =====================================================
        else:

            norm_df = df_temp[criteria].copy()

            # normalización vectorial
            for m in criteria:
                denom = (norm_df[m]**2).sum() ** 0.5
                if denom != 0:
                    norm_df[m] = norm_df[m] / denom

            ideal = {}
            anti_ideal = {}

            for m in criteria:
                if m in m_max:
                    ideal[m] = norm_df[m].max()
                    anti_ideal[m] = norm_df[m].min()
                else:
                    ideal[m] = norm_df[m].min()
                    anti_ideal[m] = norm_df[m].max()

            d_plus = []
            d_minus = []

            for i in range(len(norm_df)):
                row = norm_df.iloc[i]

                dp = sum((row[m] - ideal[m])**2 for m in criteria) ** 0.5
                dm = sum((row[m] - anti_ideal[m])**2 for m in criteria) ** 0.5

                d_plus.append(dp)
                d_minus.append(dm)

            df_temp["score_topsis"] = [
                dm / (dp + dm) if (dp + dm) != 0 else 0
                for dp, dm in zip(d_plus, d_minus)
            ]

            score_col = "score_topsis"

        # -------------------------------
        # Top N
        # -------------------------------
        n = st.sidebar.slider(
            "Top N",
            1,
            len(df_temp),
            min(10, len(df_temp)),
            disabled=lens_locked
        )

        selected_df = df_temp.sort_values(score_col, ascending=False).head(n)
        color_col = score_col

# ----------------------------------
# DIVERSITY METHODS
# ----------------------------------

elif mode == "Clustering":

    st.sidebar.markdown("### Clustering")

    # -------------------------------
    # Selección de método
    # -------------------------------
    cluster_method = st.sidebar.radio(
        "Clustering method",
        ["K-Medoids", "HDBSCAN"]
    )

    # -------------------------------
    # Selección de métricas
    # -------------------------------
    cluster_metrics = st.sidebar.multiselect(
        "Metrics for clustering",
        available_metrics,
        default=available_metrics[:2]
    )

    if cluster_metrics and len(cluster_metrics) >= 2:

        # -------------------------------
        # Preparar datos
        # -------------------------------
        X = selected_df[cluster_metrics].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ==========================================================
        # ✅ CASO 1: K-MEDOIDS
        # ==========================================================
        if cluster_method == "K-Medoids":
            k_mode = st.sidebar.radio(
                "Number of clusters",
                ["Manual", "Auto"]
            )
            if k_mode == "Manual":
                k = st.sidebar.slider(
                    "k clusters",
                    2,
                    min(10, len(selected_df)),
                    3
                )

            else:
                best_k = 2
                best_score = -1

                for k_test in range(2, min(10, len(X_scaled))):
                    try:
                        model = KMedoids(
                            n_clusters=k_test,
                            method='pam',
                            random_state=123
                        )
                        labels_test = model.fit_predict(X_scaled)

                        if len(set(labels_test)) > 1:
                            score = silhouette_score(X_scaled, labels_test)

                            if score > best_score:
                                best_score = score
                                best_k = k_test
                    except:
                        pass

                k = best_k
                st.sidebar.info(f"Suggested k : {k}")

            model = KMedoids(
                n_clusters=k,
                method='pam',
                random_state=123
            )
            labels = model.fit_predict(X_scaled)

        # ==========================================================
        # ✅ CASO 2: HDBSCAN  
        # ==========================================================
        else:

            st.sidebar.markdown("#### HDBSCAN configuration")

            mode_size = st.sidebar.radio(
                "Cluster size",
                ["Auto (recommended)", "Manual"]
            )

            N = len(selected_df)

            if mode_size == "Auto (recommended)":

                option = st.sidebar.selectbox(
                    "Cluster granularity",
                    ["Small (~5%)", "Medium (~10%)", "Large (~20%)"],
                    index=1
                )

                if option == "Small (~5%)":
                    min_cluster_size = max(2, int(0.05 * N))
                elif option == "Medium (~10%)":
                    min_cluster_size = max(2, int(0.10 * N))
                else:
                    min_cluster_size = max(2, int(0.20 * N))

                st.sidebar.info(f"min_cluster_size = {min_cluster_size}")

            else:
                min_cluster_size = st.sidebar.slider(
                    "Min cluster size",
                    2,
                    len(selected_df),
                    max(2, int(0.1 * N))
                )

            model = HDBSCAN(
                min_cluster_size=min_cluster_size
            )

            labels = model.fit_predict(X_scaled)


        # -------------------------------
        # Métricas de clustering (info)
        # -------------------------------
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).sum() / len(labels)

        st.sidebar.info(f"Clusters: {n_clusters} | Noise: {noise_ratio:.2%}")


        # -------------------------------
        # Guardar resultados (común)
        # -------------------------------
        selected_df["cluster"] = labels

        # Convertir a string para visualización
        selected_df["cluster_str"] = selected_df["cluster"].astype(str)

        # Manejar ruido de HDBSCAN (-1)
        selected_df["cluster_str"] = selected_df["cluster_str"].replace("-1", "Noise")

        # Tamaño por grupo
        cluster_sizes = selected_df.groupby("cluster_str")["id"].transform("size")

        selected_df["group_label"] = (
            "Cluster " +
            selected_df["cluster_str"] +
            " (n=" +
            cluster_sizes.astype(str) +
            ")"
        )

        color_col = "group_label"

#--------------------------------------
#   Efficiency based Ratio
#-------------------------------------

elif mode == "Efficiency-Ratio":

    st.sidebar.markdown("### Efficiency (Benefit/Cost)")

    benefit = st.sidebar.selectbox(
        "Benefit (maximize)", 
        available_qual, 
        key="eff_benefit"
    )

    cost = st.sidebar.selectbox(
        "Cost (minimize)", 
        available_metrics, 
        key="eff_cost"
    )

    # protección
    if benefit == cost:
        st.warning("Benefit and Cost must be different metrics")
        st.stop()

    # slider primero

    n = st.sidebar.slider(
        "Top N efficient solutions",
        1,
        len(selected_df),
        min(10, len(selected_df)),
        key="eff_topn"
    )

    # calcular score
    selected_df = selected_df.copy()

    cost_safe = selected_df[cost].replace(0, 1e-9)
    selected_df["efficiency_score"] = selected_df[benefit] / cost_safe

    # ordenar
    selected_df = selected_df.sort_values("efficiency_score", ascending=False).head(n)

    color_col = "efficiency_score"

# --------------------------------------------
# RANKING BASED (count SIN NORMALIZAR)
# --------------------------------------------


elif mode == "Ranking-based":
    sel_metrics = st.sidebar.multiselect("Quality metrics", available_qual)
    n_top = st.sidebar.slider("Top N per metric", 1, min(50, len(selected_df)), 10)

    if sel_metrics:
        ranks = []
        goals = {}

        for m in sel_metrics:
            goal = st.sidebar.selectbox(
                f"Goal for {m}",
                ["Maximize", "Minimize"],
                key=f"g_{m}"
            )
            goals[m] = goal
            # ✅ usa selected_df (= framing actual), no filtered_df directamente,
            # para que esta lente nunca pueda "ver" nada fuera del framing
            # aunque el orden del código cambie en el futuro.
            ranks.append(
                selected_df.sort_values(m, ascending=(goal == "Minimize")).head(n_top)
            )

        counts = pd.concat(ranks).groupby("id").size().reset_index(name="count")

        # Mantener todo el subconjunto actual y marcar count=0 si no aparece en ningún top-N
        selected_df = selected_df.merge(counts, on="id", how="left").fillna(0)
        selected_df["count"] = selected_df["count"].astype(int)

        # Etiqueta base sin conteo agregado todavía
        selected_df["group_base"] = selected_df["count"].apply(
            lambda c: "No match" if c == 0 else f"Matches = {c}"
        )

        # Añadir el n de cada grupo para que salga en la leyenda
        group_sizes = selected_df["group_base"].value_counts().to_dict()
        selected_df["group_label"] = selected_df["group_base"].apply(
            lambda g: f"{g} (n={group_sizes[g]})"
        )

        color_col = "group_label"
        threshold = max(1, len(sel_metrics) - 1)


# --------------------------------------------
# NONE
# --------------------------------------------
else:
    color_col = None

# Guardar explícitamente el subconjunto actual derivado de la lente ROI
roi_df = selected_df.copy()

# --------------------------------------------
# HIGHLIGHT + LABELS
# --------------------------------------------

# Guardamos explícitamente el subconjunto actual derivado del framing + lens.
# Esto representa la ROI/subset estructural actual antes de aplicar foco manual.
roi_df = selected_df.copy()

# --------------------------------------------
# ⚠️ SANEAR HIGHLIGHT ANTES DE CREAR EL WIDGET
# --------------------------------------------
# Cada cambio de framing o de lens puede reducir/cambiar roi_df. Si algún ID
# marcado anteriormente ya no existe en la ROI actual, el valor persistido en
# st.session_state["selected_ids"] queda "huérfano" respecto a las nuevas
# `options` del multiselect, y Streamlit puede fallar o comportarse de forma
# inconsistente al re-renderizar el widget con la misma key.
#
# Por definición del workflow (CSS ⊆ SOI), un candidato resaltado que ya no
# pertenece a la SOI actual deja de tener sentido como highlight: lo quitamos
# de forma explícita ANTES de instanciar el widget, y avisamos al usuario.



# --------------------------------------------
# SOI FOCUS + COMPARATIVE SUPPORT
# --------------------------------------------


focus_mode = st.sidebar.checkbox(
    "Lock current candidate set",
    help="Keep candidates highlighted in context, or restrict analysis to them.",
    key="focus_mode"
)


if focus_mode:
    st.session_state.focus_locked = True
else:
    st.session_state.focus_locked = False

focus_group = "All"

if focus_mode:

    grouping_col = None

    if "cluster_str" in roi_df.columns:
        grouping_col = "cluster_str"

    elif "group_label" in roi_df.columns:
        grouping_col = "group_label"

    if grouping_col is not None:

        groups = sorted(
            roi_df[grouping_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        focus_group = st.sidebar.selectbox(
            "Focus group",
            ["All"] + groups,
            key="focus_group"
        )


active_soi = None


if focus_mode:

    st.caption(
        "📚 Load saved SOI disabled while focus is active"
    )

    st.session_state["active_soi"] = "None"

else:

    if st.session_state.saved_sois:

        soi_names = [
            soi["name"]
            for soi in st.session_state.saved_sois
        ]

        # Evitar que Streamlit conserve una selección inválida
        if (
            st.session_state.get("active_soi")
            not in ["None"] + soi_names
        ):
            st.session_state["active_soi"] = "None"

        active_soi = st.selectbox(
            "📚 Load saved SOI",
            ["None"] + soi_names,
            key="active_soi"
        )

# ----------------------------------
# Apply active combined SOI
# ----------------------------------
# ----------------------------------
# Apply active combined SOI
# ----------------------------------
if st.session_state.active_combined_soi is not None:
    combined_soi = st.session_state.active_combined_soi
    combined_ids = combined_soi.get("ids", [])

    roi_df = roi_df[
        roi_df["id"].isin(combined_ids)
    ].copy()

    consensus_scores = combined_soi.get("consensus_scores", {})
    roi_df["consensus_score"] = roi_df["id"].map(consensus_scores).fillna(0)

    color_col = "consensus_score"

    st.sidebar.info(
        f"Active combined SOI: {combined_soi['name']} "
        f"[{len(combined_ids)} solutions]"
    )

    if st.sidebar.button(
        "Clear active combined SOI",
        key="clear_active_combined_soi"
    ):
        st.session_state.active_combined_soi = None
        st.session_state.pending_clear_selected_ids = True
        st.rerun()



# ----------------------------------
# Recompute valid IDs after saved/combined SOI filtering
# ----------------------------------
valid_roi_ids = roi_df["id"].tolist()

# ----------------------------------
# Pending clear must happen before the widget is created
# ----------------------------------
if st.session_state.pending_clear_selected_ids:
    st.session_state.selected_ids = []
    st.session_state.pending_clear_selected_ids = False

# ----------------------------------
# Remove highlighted candidates that are no longer in the active SOI
# ----------------------------------
previous_selected_ids = st.session_state.get("selected_ids", [])
dropped_ids = [
    sid for sid in previous_selected_ids
    if sid not in valid_roi_ids
]

if dropped_ids:
    st.session_state.selected_ids = [
        sid for sid in previous_selected_ids
        if sid in valid_roi_ids
    ]

    st.sidebar.warning(
        f"{len(dropped_ids)} previously highlighted solution(s) fell outside "
        "the current framing/lens/SOI and were unhighlighted: "
        f"{', '.join(str(int(i)) for i in dropped_ids)}"
    )

# ----------------------------------
# Highlight widget
# ----------------------------------
selected_ids = st.multiselect(
    " 👆 Highlight candidate solutions",
    options=valid_roi_ids,
    key="selected_ids",
    help="Manually mark solutions for visual tracking or focused analysis."
)


roi_df["highlight"] = roi_df["id"].isin(selected_ids)

if show_ids:
    if mode == "Ranking-based":
        roi_df["label"] = roi_df.apply(
            lambda r: str(int(r["id"])) if (r["highlight"] or int(r.get("count", 0)) >= threshold) else "",
            axis=1
        )
    else:
        roi_df["label"] = roi_df["id"].astype(str)
else:
    roi_df["label"] = ""



if st.session_state.focus_locked:

    st.sidebar.success(
        "🔒 Candidate set locked"
    )

    if (
        "cluster_str" in roi_df.columns
        and focus_group != "All"
    ):
        st.sidebar.caption(
            f"Focused cluster: {focus_group}"
        )

    elif (
        "group_label" in roi_df.columns
        and focus_group != "All"
    ):
        st.sidebar.caption(
            f"Focused group: {focus_group}"
        )

    default_soi_name = (
        focus_group
        if focus_group != "All"
        else f"SOI {len(st.session_state.saved_sois)+1}"
    )

    col_name, col_save = st.sidebar.columns([0.75, 0.25])

    with col_name:

        soi_name = st.text_input(
            "SOI name",
            value=default_soi_name,
            label_visibility="collapsed",
            key="soi_name_input"
        )

    with col_save:

        save_soi = st.button(
            "💾",
            disabled=len(st.session_state.saved_sois) >= 10,
            use_container_width=True
        )

    if save_soi:

        current_ids = st.session_state.get(
            "current_soi_ids",
            selected_df["id"].tolist()
        )

        st.sidebar.write(
            "Saving:",
            len(current_ids)
        )

        existing_names = [soi["name"] for soi in st.session_state.saved_sois]

        if soi_name in existing_names:
            st.sidebar.warning("A SOI with this name already exists. Please choose another name.")
        else:
            st.session_state.saved_sois.append(
                {
                    "name": soi_name,
                    "ids": current_ids,
                    "size": len(current_ids),
                    "source_lens": mode,
                }
            )
            st.sidebar.success(f"Saved SOI: {soi_name}")

    # ----------------------------------
    # Saved SOIs
    # ----------------------------------

    if st.session_state.saved_sois:
        st.sidebar.markdown("### 📚 Saved SOIs")

        for idx, soi in enumerate(st.session_state.saved_sois):
            size = soi.get("size", len(soi.get("ids", [])))
            source_lens = soi.get("source_lens", "unknown")

            col_info, col_del = st.sidebar.columns([0.82, 0.18])

            with col_info:
                st.caption(
                    f"• {soi['name']} [{size} solutions] · {source_lens}"
                )

            with col_del:
                delete_clicked = st.button(
                    "🗑️",
                    key=f"delete_soi_{idx}",
                    help=f"Delete SOI: {soi['name']}"
                )

            if delete_clicked:
                deleted_name = st.session_state.saved_sois[idx]["name"]
                st.session_state.saved_sois.pop(idx)

                # Si el SOI borrado estaba cargado, resetear selector
                if st.session_state.get("active_soi") == deleted_name:
                    st.session_state["active_soi"] = "None"

                # También limpiar selección de combinación si estaba marcada
                st.session_state.selected_sois_for_combination = [
                    name for name in st.session_state.selected_sois_for_combination
                    if name != deleted_name
                ]

                st.rerun()


# ----------------------------------
# SOI Combination
# ----------------------------------
if st.session_state.saved_sois:
    st.sidebar.markdown("### 🔗 SOI Combination")

    st.session_state.combine_sois_mode = st.sidebar.checkbox(
        "Combine saved SOIs",
        value=st.session_state.combine_sois_mode,
        help="Select several saved SOIs to generate a consensus SOI."
    )

    if st.session_state.combine_sois_mode:
        st.sidebar.caption("Select SOIs to include in the combination:")

        selected_for_combination = []

        for idx, soi in enumerate(st.session_state.saved_sois):
            soi_name = soi["name"]
            size = soi.get("size", len(soi.get("ids", [])))

            checked = st.sidebar.checkbox(
                f"{soi_name} [{size}]",
                value=soi_name in st.session_state.selected_sois_for_combination,
                key=f"combine_soi_{idx}_{soi_name}"
            )

            if checked:
                selected_for_combination.append(soi_name)

        st.session_state.selected_sois_for_combination = selected_for_combination

        st.sidebar.info(
            f"{len(selected_for_combination)} SOI(s) selected for combination"
        )

        consensus_threshold = st.sidebar.slider(
            "Consensus level",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.consensus_threshold,
            step=0.05,
            help=(
                "Minimum proportion of selected SOIs that must support a solution. "
                "Higher values produce a smaller consensus core; lower values produce a broader consensus pool."
            ),
            key="consensus_threshold_slider"
        )

        st.session_state.consensus_threshold = consensus_threshold

        if consensus_threshold >= 0.75:
            st.sidebar.caption("Mode: consensus core")
        elif consensus_threshold >= 0.50:
            st.sidebar.caption("Mode: consensus pool")
        else:
            st.sidebar.caption("Mode: broad exploratory pool")

        col_apply, col_clear = st.sidebar.columns(2)

        with col_apply:
            apply_combine = st.button(
                "Apply",
                key="apply_soi_combination",
                use_container_width=True,
                disabled=len(selected_for_combination) < 2
            )

        with col_clear:
            clear_combine = st.button(
                "Clear",
                key="clear_soi_combination",
                use_container_width=True
            )

        if clear_combine:
            st.session_state.selected_sois_for_combination = []
            st.session_state.active_combined_soi = None
            st.session_state.pending_clear_selected_ids = True
            st.rerun()

        if apply_combine:
            selected_sois = [
                soi for soi in st.session_state.saved_sois
                if soi["name"] in selected_for_combination
            ]

            n_sois = len(selected_sois)

            support = {}
            support_names = {}

            for soi in selected_sois:
                soi_name = soi["name"]
                unique_ids = set(soi.get("ids", []))

                for sid in unique_ids:
                    support[sid] = support.get(sid, 0) + 1
                    support_names.setdefault(sid, []).append(soi_name)

            consensus_scores = {
                sid: support_count / n_sois
                for sid, support_count in support.items()
            }

            combined_ids = [
                sid for sid, score in consensus_scores.items()
                if score >= consensus_threshold
            ]

            combined_ids = sorted(combined_ids)

            if len(combined_ids) == 0:
                st.sidebar.warning(
                    "No solutions satisfy the selected consensus level. Try lowering the threshold."
                )
            else:
                st.session_state.active_combined_soi = {
                    "name": f"Consensus SOI ({consensus_threshold:.2f})",
                    "ids": combined_ids,
                    "size": len(combined_ids),
                    "threshold": consensus_threshold,
                    "source_sois": selected_for_combination,
                    "consensus_scores": consensus_scores,
                    "support_names": support_names,
                }

                st.session_state.pending_reset_active_soi = True
                st.session_state.pending_clear_selected_ids = True

                st.sidebar.success(
                    f"Consensus SOI applied: {len(combined_ids)} solutions"
                )

                st.rerun()


# ----------------------------------
# Detailed comparison activation
# ----------------------------------
st.sidebar.markdown("## 🎯 Candidate Solution Set focus and Comparison")

st.sidebar.checkbox(
    "Open detailed comparison",
    key="show_comparison"
)


selected_df = roi_df.copy()

if focus_mode:

    # ----------------------------------
    # 1. Focus group (cluster / ranking)
    # ----------------------------------

    if "cluster_str" in roi_df.columns:

        if focus_group != "All":

            selected_df = roi_df[
                roi_df["cluster_str"].astype(str)
                == str(focus_group)
            ].copy()

    elif "group_label" in roi_df.columns:

        if focus_group != "All":

            selected_df = roi_df[
                roi_df["group_label"].astype(str)
                == str(focus_group)
            ].copy()

    st.session_state["current_soi_ids"] = (
        selected_df["id"].tolist()
    )


    # ----------------------------------
    # 2. Highlight manual (refinamiento)
    # ----------------------------------

    if selected_df["highlight"].any():

        selected_df = selected_df[
            selected_df["highlight"]
        ].copy()

        st.sidebar.caption(
            "Manual refinement applied"
        )

    else:

        st.sidebar.caption(
            "No manual refinement applied"
        )

    # ----------------------------------
    # 3. Tamaño final del subconjunto
    # ----------------------------------

    st.sidebar.info(
        f"Focused subset size: {len(selected_df)} solutions"
    )



# --------------------------------------------
# GRÁFICOS
# --------------------------------------------

for i, group in enumerate(st.session_state.groups):
    st.subheader(f"Decision-Space Map {i+1}")
    
    others = [m for idx, g in enumerate(st.session_state.groups) if idx != i for m in g if m]
    available_here = [m for m in available_metrics if m not in others]

    if len(available_here) < 2:
        st.warning("No more metrics available for this group.")
        continue

    col1, col2, col3 = st.columns(3)
    
    with col1:
        curr_x = group[0] if group[0] in available_here else available_here[0]
        x = st.selectbox(
            f"X Axis {i}",
            available_here,
            index=available_here.index(curr_x),
            key=f"x_{i}"
        )
    
    with col2:
        y_opts = [m for m in available_here if m != x]
        curr_y = group[1] if group[1] in y_opts else y_opts[0]
        y = st.selectbox(
            f"Y Axis {i}",
            y_opts,
            index=y_opts.index(curr_y),
            key=f"y_{i}"
        )
    
    with col3:
        s_opts = [None] + [m for m in available_here if m not in [x, y]]
        curr_s = group[2] if group[2] in s_opts else None
        size = st.selectbox(
            f"Size {i}",
            s_opts,
            index=s_opts.index(curr_s),
            key=f"s_{i}"
        )

    st.session_state.groups[i] = [x, y, size]

    df_plot = selected_df.copy()

    cA, cB = st.columns(2)
    with cA:
        render_scatter_plot(df_plot, x, y, None, color_col, show_ids, key=f"p1_{i}")
    with cB:
        if size:
            render_scatter_plot(df_plot, x, size, y, color_col, show_ids, key=f"p2_{i}")
        else:
            st.info("Select a metric in 'Size' to enable comparison.")

# --------------------------------------------
# RADAR / DETAILED COMPARISON
# --------------------------------------------

if st.session_state.show_comparison:

    st.markdown("### 🆚 Detailed comparison")

    # Si el usuario ha activado focus y hay highlights,
    if focus_mode and roi_df["highlight"].any():
        df_compare_base = selected_df.copy()
        st.caption("Comparison source: focused candidate set")

    else:
        compare_options = ["Current subset"]

        if roi_df["highlight"].any():
            compare_options.append("Highlighted candidates")

        # ⚠️ Mismo problema que con selected_ids: si la opción elegida
        # anteriormente ("Highlighted candidates") ya no está disponible
        # (p. ej. porque el highlight se vació al cambiar el framing/lens),
        # el valor persistido en session_state["comparison_source"] queda
        # huérfano respecto a las nuevas `compare_options`. Lo saneamos
        # antes de crear el widget.
        if st.session_state.get("comparison_source") not in compare_options:
            st.session_state["comparison_source"] = compare_options[0]

        compare_source = st.radio(
            "Comparison source",
            compare_options,
            horizontal=True,
            key="comparison_source"
        )

        if compare_source == "Highlighted candidates":
            df_compare_base = roi_df[roi_df["highlight"] == True].copy()
        else:
            df_compare_base = roi_df.copy()

    # Detectar columna de agrupación (cluster o count)
    group_col = None

    if "cluster_str" in df_compare_base.columns:
        group_col = "cluster_str"
    elif "cluster" in df_compare_base.columns:
        group_col = "cluster"

    if group_col is None:

        if "group_label" in df_compare_base.columns:
            group_col = "group_label"

        elif "count" in df_compare_base.columns:
            group_col = "count"


    plot_radar(df_compare_base, available_metrics, group_col=group_col)


if not st.session_state.show_comparison:
    st.session_state.pop(
        "selected_group_export",
        None
    )

    st.session_state.pop(
        "selected_group_column",
        None
    )
# --------------------------------------------
# CURRENT DECISION SUBSET + EXPORT
# --------------------------------------------

col_titulo, col_btn = st.columns([3, 1], vertical_alignment="center")

with col_titulo:
    st.markdown("📋 Current decision subset")

export_df = selected_df.copy()

selected_group = st.session_state.get(
    "selected_group_export",
    "All"
)

selected_group_col = st.session_state.get(
    "selected_group_column"
)

if (
    selected_group != "All"
    and selected_group_col is not None
    and selected_group_col in export_df.columns
):
    export_df = export_df[
        export_df[selected_group_col].astype(str)
        == str(selected_group)
    ]

#st.write("selected_group =", selected_group)
#st.write("selected_group_col =", selected_group_col)
#st.write("rows export =", len(export_df))

csv_data = export_df.drop(
    columns=["highlight", "label"],
    errors="ignore"
)







with col_btn:
    st.download_button(
        label="⬇️ Export current subset",
        data=csv_data.to_csv(index=False),
        file_name="current_subset.csv",
        mime="text/csv",
        use_container_width=True
    )

# --------------------------------------------
# PREVIEW
# --------------------------------------------
with st.expander("Preview", expanded=False):
    df_preview = export_df.copy()
    
    # 1. Si "id" existe, lo ponemos como índice antes de borrar las demás columnas
    if "id" in df_preview.columns:
        df_preview = df_preview.set_index("id")
        
    # 2. Ocultamos el resto de columnas de control (quitamos "id" de esta lista)
    columnas_a_ocultar = ["highlight", "highlight_label", "label"]
    df_preview = df_preview.drop(
        columns=[col for col in columnas_a_ocultar if col in df_preview.columns]
    )
    
    st.dataframe(df_preview.head(100), use_container_width=True)


# ----------------------------------
# 🥇 Top solutions (non-intrusive)
# ----------------------------------
#if mode in ["MCDM", "Efficiency-Ratio"] and len(selected_df) >= 1:
#    top_n = min(3, len(selected_df))
#    top_ids = selected_df.head(top_n)["id"].astype(int).tolist()
#    st.caption("### 🥇 Top solutions (current method)")
#    st.caption(", ".join([f"ID {i}" for i in top_ids]))

# ----------------------------------
# 📊 Quick insights (non-intrusive)
# ----------------------------------
#if len(selected_df) >= 2:
#    numeric_cols = selected_df.select_dtypes(include="number").columns.tolist()
#    exclude_cols = ["id"]
#    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
#    if numeric_cols:
#        means = selected_df[numeric_cols].mean()
#        top_metrics = means.sort_values(ascending=False).head(3)
#        items = [f"**{m}**: {v:.3f}" for m, v in top_metrics.items()]
#        st.markdown("📊 Quick insights:  "+" | ".join(items))
