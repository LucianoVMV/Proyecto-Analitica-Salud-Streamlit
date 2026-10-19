from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Estimación de monto liquidado",
    page_icon="🏥",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

RUTA_CONFIG = BASE / "models" / "config_modelo.json"
RUTA_MODELO = BASE / "models" / "modelo_catboost.cbm"
RUTA_DATOS = BASE / "data" / "datos_procesados.parquet"


# ============================================================
# 2. CARGAR CONFIGURACIÓN
# ============================================================

with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

PREDICTORES = CONFIG["predictores"]
CATEGORICAS = CONFIG["categoricas"]
VALOR_NULO = CONFIG["valor_nulo_categoricas"]


# Diccionario:
# variable -> codigo -> etiqueta

ETIQUETAS = {}

for variable, categorias in CONFIG["categorias_permitidas"].items():
    ETIQUETAS[variable] = {
        str(item["codigo"]): str(item["etiqueta"])
        for item in categorias
    }


def opciones(variable):
    return list(ETIQUETAS[variable].keys())


def mostrar(variable, codigo):
    """Texto que verá el usuario en los selectbox."""
    codigo = str(codigo)
    etiqueta = ETIQUETAS[variable].get(codigo, codigo)

    if etiqueta == codigo:
        return codigo

    return f"{etiqueta} [{codigo}]"


def indice_default(lista, valor):
    valor = str(valor)
    return lista.index(valor) if valor in lista else 0


# ============================================================
# 3. CARGAR MODELO Y DATOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    modelo = CatBoostRegressor()
    modelo.load_model(RUTA_MODELO)
    return modelo


@st.cache_data
def cargar_datos():
    columnas = [
        "PRIMER_DIAGNOSTICO",
        "DIAG_CAPITULO",
        "CODIGO_IAFAS_CRIPTO",
        "CODIGO_IPRESS_CRIPTO",
        "UBIGEO_IPRESS",
        "DEPTO_IPRESS_COD",
    ]

    df = pd.read_parquet(RUTA_DATOS, columns=columnas)

    for c in columnas:
        df[c] = df[c].astype("string").fillna(VALOR_NULO)

    return df


modelo = cargar_modelo()
datos = cargar_datos()


# ============================================================
# 4. CREAR MAPAS AUXILIARES
# ============================================================

def crear_mapa_moda(df, clave, valor):
    """
    Para cada valor de 'clave', devuelve el valor de 'valor'
    más frecuente.
    """
    temp = (
        df[[clave, valor]]
        .dropna()
        .value_counts()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates(subset=[clave])
    )

    return dict(zip(temp[clave].astype(str), temp[valor].astype(str)))


# Diagnóstico -> capítulo CIE
MAPA_DIAG_CAP = crear_mapa_moda(
    datos,
    "PRIMER_DIAGNOSTICO",
    "DIAG_CAPITULO"
)

# IPRESS -> ubigeo
MAPA_IPRESS_UBIGEO = crear_mapa_moda(
    datos,
    "CODIGO_IPRESS_CRIPTO",
    "UBIGEO_IPRESS"
)

# IPRESS -> departamento INEI
MAPA_IPRESS_DEPTO = crear_mapa_moda(
    datos,
    "CODIGO_IPRESS_CRIPTO",
    "DEPTO_IPRESS_COD"
)


# ============================================================
# 5. INTERFAZ
# ============================================================

st.title("Estimación del monto liquidado por atención")

st.write(
    """
    Prototipo analítico para estimar el monto liquidado esperado
    de una atención de salud utilizando el modelo CatBoost
    desarrollado en el proyecto.
    """
)

st.info(
    "El resultado es una referencia analítica. "
    "No representa el monto correcto de una liquidación ni sustituye "
    "la revisión profesional."
)


# ============================================================
# COBERTURA Y DIAGNÓSTICO
# ============================================================

st.subheader("1. Características de la atención")

col1, col2, col3 = st.columns(3)

with col1:

    ops_cobertura = opciones("TIPO_COBERTURA")

    tipo_cobertura = st.selectbox(
        "Tipo de cobertura",
        ops_cobertura,
        index=indice_default(ops_cobertura, "4"),
        format_func=lambda x: mostrar("TIPO_COBERTURA", x)
    )

    ops_subtipo = opciones("SUBTIPO_COBERTURA")

    subtipo_cobertura = st.selectbox(
        "Subtipo de cobertura",
        ops_subtipo,
        index=indice_default(ops_subtipo, "100"),
        format_func=lambda x: mostrar("SUBTIPO_COBERTURA", x)
    )

    ops_diag = opciones("PRIMER_DIAGNOSTICO")

    diagnostico = st.selectbox(
        "Diagnóstico principal (CIE-10)",
        ops_diag,
        index=indice_default(ops_diag, "H16.2"),
        format_func=lambda x: mostrar("PRIMER_DIAGNOSTICO", x)
    )

    # Se obtiene automáticamente a partir del diagnóstico
    diag_capitulo = MAPA_DIAG_CAP.get(
        diagnostico,
        opciones("DIAG_CAPITULO")[0]
    )

    st.text_input(
        "Capítulo CIE-10",
        value=mostrar("DIAG_CAPITULO", diag_capitulo),
        disabled=True
    )


# ============================================================
# PACIENTE
# ============================================================

with col2:

    ops_prof = opciones("TIPO_PROFESIONAL")

    tipo_profesional = st.selectbox(
        "Tipo de profesional",
        ops_prof,
        index=indice_default(ops_prof, "01"),
        format_func=lambda x: mostrar("TIPO_PROFESIONAL", x)
    )

    ops_afiliacion = opciones("TIPO_AFILIACION_DEL_PACIENTE")

    tipo_afiliacion = st.selectbox(
        "Tipo de afiliación",
        ops_afiliacion,
        index=indice_default(ops_afiliacion, "1"),
        format_func=lambda x: mostrar(
            "TIPO_AFILIACION_DEL_PACIENTE", x
        )
    )

    ops_filiacion = opciones("FILIACION_PACIENTE")

    filiacion = st.selectbox(
        "Filiación del paciente",
        ops_filiacion,
        index=indice_default(ops_filiacion, "01"),
        format_func=lambda x: mostrar("FILIACION_PACIENTE", x)
    )

    ops_sexo = opciones("SEXO_PACIENTE")

    sexo = st.selectbox(
        "Sexo del paciente",
        ops_sexo,
        index=indice_default(ops_sexo, "2"),
        format_func=lambda x: mostrar("SEXO_PACIENTE", x)
    )

    edad = st.number_input(
        "Edad",
        min_value=0,
        max_value=100,
        value=48,
        step=1
    )


# ============================================================
# ASEGURADORA / PRESTADOR
# ============================================================

with col3:

    ops_iafas = opciones("CODIGO_IAFAS_CRIPTO")

    IAFAS_EJEMPLO = "52C8E6961E279AA6459E8782F8F316B9220B782A"

    iafas = st.selectbox(
        "IAFAS",
        ops_iafas,
        index=indice_default(ops_iafas, IAFAS_EJEMPLO),
        format_func=lambda x: mostrar("CODIGO_IAFAS_CRIPTO", x)
    )

    ops_ipress = opciones("CODIGO_IPRESS_CRIPTO")

    IPRESS_EJEMPLO = "7BF2CF97940320F1B9781DBEA758928CD3BF3425"

    ipress = st.selectbox(
        "IPRESS",
        ops_ipress,
        index=indice_default(ops_ipress, IPRESS_EJEMPLO),
        format_func=lambda x: mostrar("CODIGO_IPRESS_CRIPTO", x)
    )

    # Se derivan automáticamente de la IPRESS
    ubigeo_ipress = MAPA_IPRESS_UBIGEO.get(
        ipress,
        opciones("UBIGEO_IPRESS")[0]
    )

    depto_ipress = MAPA_IPRESS_DEPTO.get(
        ipress,
        opciones("DEPTO_IPRESS_COD")[0]
    )

    st.text_input(
        "Ubigeo de la IPRESS",
        value=ubigeo_ipress,
        disabled=True
    )

    st.text_input(
        "Departamento de la IPRESS",
        value=mostrar("DEPTO_IPRESS_COD", depto_ipress),
        disabled=True
    )


# ============================================================
# 6. DATOS ADICIONALES
# ============================================================

st.subheader("2. Datos adicionales")

c1, c2, c3 = st.columns(3)

with c1:

    ops_dep_af = opciones("DEPTO_AFILIADO_COD")

    depto_afiliado = st.selectbox(
        "Departamento del afiliado",
        ops_dep_af,
        index=indice_default(ops_dep_af, "14"),
        format_func=lambda x: mostrar("DEPTO_AFILIADO_COD", x)
    )

with c2:

    ops_pago = opciones("TIPO_DOC_PAGO")

    tipo_doc_pago = st.selectbox(
        "Tipo de documento de pago",
        ops_pago,
        index=indice_default(ops_pago, "01"),
        format_func=lambda x: mostrar("TIPO_DOC_PAGO", x)
    )

with c3:

    ops_identidad = opciones("TIPO_DOC_IDENTIDAD")

    tipo_doc_identidad = st.selectbox(
        "Tipo de documento de identidad",
        ops_identidad,
        index=indice_default(ops_identidad, "1"),
        format_func=lambda x: mostrar("TIPO_DOC_IDENTIDAD", x)
    )


# ============================================================
# 7. CALCULAR MISMO DEPARTAMENTO
# ============================================================

nombre_dep_ipress = ETIQUETAS["DEPTO_IPRESS_COD"].get(
    str(depto_ipress),
    str(depto_ipress)
)

nombre_dep_afiliado = ETIQUETAS["DEPTO_AFILIADO_COD"].get(
    str(depto_afiliado),
    str(depto_afiliado)
)

if (
    str(depto_ipress) == VALOR_NULO
    or str(depto_afiliado) == VALOR_NULO
):
    mismo_depto = VALOR_NULO
else:
    # Se comparan los nombres, no los códigos,
    # porque INEI y RENIEC usan códigos distintos.
    mismo_depto = str(
        nombre_dep_ipress == nombre_dep_afiliado
    )


st.caption(
    f"Coincidencia geográfica: "
    f"{nombre_dep_afiliado} / {nombre_dep_ipress} → "
    f"MISMO_DEPTO = {mismo_depto}"
)


# ============================================================
# 8. MONTO OBSERVADO OPCIONAL
# ============================================================

st.subheader("3. Comparación opcional")

comparar = st.checkbox(
    "Tengo un monto liquidado observado y quiero compararlo"
)

monto_observado = None

if comparar:
    monto_observado = st.number_input(
        "Monto liquidado observado (S/)",
        min_value=0.0,
        value=163.03,
        step=10.0,
        format="%.2f"
    )


# ============================================================
# 9. PREDICCIÓN
# ============================================================

st.divider()

if st.button(
    "Estimar monto",
    type="primary",
    use_container_width=True
):

    entrada = pd.DataFrame([{
        "TIPO_COBERTURA": tipo_cobertura,
        "SUBTIPO_COBERTURA": subtipo_cobertura,
        "PRIMER_DIAGNOSTICO": diagnostico,
        "DIAG_CAPITULO": diag_capitulo,
        "TIPO_PROFESIONAL": tipo_profesional,
        "TIPO_AFILIACION_DEL_PACIENTE": tipo_afiliacion,
        "FILIACION_PACIENTE": filiacion,
        "SEXO_PACIENTE": sexo,
        "UBIGEO_IPRESS": ubigeo_ipress,
        "DEPTO_IPRESS_COD": depto_ipress,
        "DEPTO_AFILIADO_COD": depto_afiliado,
        "MISMO_DEPTO": mismo_depto,
        "CODIGO_IAFAS_CRIPTO": iafas,
        "CODIGO_IPRESS_CRIPTO": ipress,
        "TIPO_DOC_PAGO": tipo_doc_pago,
        "TIPO_DOC_IDENTIDAD": tipo_doc_identidad,
        "EDAD": float(edad)
    }])

    # Exactamente el mismo orden del entrenamiento
    entrada = entrada[PREDICTORES]

    # CatBoost espera las categóricas como cadenas
    for columna in CATEGORICAS:
        entrada[columna] = (
            entrada[columna]
            .fillna(VALOR_NULO)
            .astype(str)
        )

    entrada["EDAD"] = pd.to_numeric(
        entrada["EDAD"],
        errors="coerce"
    )

    # El modelo predice log1p(monto)
    pred_log = float(modelo.predict(entrada)[0])

    # Configuración del proyecto:
    # se recorta log a >= 0 y se vuelve a soles con expm1
    pred_log = max(0.0, pred_log)

    monto_predicho = float(np.expm1(pred_log))

    st.subheader("Resultado")

    st.metric(
        "Monto liquidado esperado",
        f"S/ {monto_predicho:,.2f}"
    )

    # Métricas globales del modelo
    metricas = CONFIG["metricas_test"]

    st.caption(
        f"Referencia del conjunto de prueba: "
        f"MedAE = S/ {metricas['MedAE_soles']:,.2f} | "
        f"MdAPE = {metricas['MdAPE_pct']:.2f}%"
    )

    # ========================================================
    # COMPARACIÓN CON MONTO REAL
    # ========================================================

    if comparar and monto_observado is not None:

        diferencia = monto_observado - monto_predicho
        diferencia_abs = abs(diferencia)

        if monto_observado > 0:
            error_pct = (
                diferencia_abs / monto_observado
            ) * 100
        else:
            error_pct = np.nan

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Monto observado",
            f"S/ {monto_observado:,.2f}"
        )

        r2.metric(
            "Diferencia",
            f"S/ {diferencia:+,.2f}"
        )

        if np.isfinite(error_pct):
            r3.metric(
                "Diferencia porcentual",
                f"{error_pct:.2f}%"
            )
        else:
            r3.metric(
                "Diferencia porcentual",
                "No calculable"
            )

        if diferencia > 0:
            st.warning(
                "El monto observado se encuentra por encima "
                "de la estimación del modelo."
            )

        elif diferencia < 0:
            st.info(
                "El monto observado se encuentra por debajo "
                "de la estimación del modelo."
            )

        else:
            st.success(
                "El monto observado coincide con la estimación."
            )

        st.warning(
            "Una desviación elevada no demuestra por sí sola "
            "que una liquidación sea incorrecta. "
            "El resultado debe utilizarse como apoyo para "
            "priorizar una revisión profesional."
        )

    # Para trazabilidad / demostración
    with st.expander("Ver variables enviadas al modelo"):
        st.dataframe(
            entrada.T.rename(columns={0: "Valor"}),
            use_container_width=True
        )