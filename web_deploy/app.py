import streamlit as st
import json
import os
from groq import Groq

# --- CONFIGURACIÓN ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PARENT_DIR, "config")
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="IA Agentes Unificados", page_icon="🤖", layout="wide")

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #4a4d6d;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE BACKEND ---
def get_groq_client():
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Falta la API Key de Groq en los Secretos.")
        st.stop()
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def cargar_marcadores():
    marcadores = {}
    if not os.path.exists(CONFIG_DIR):
        return marcadores

    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(CONFIG_DIR, filename), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    marcadores[str(data["id"])] = data
                except Exception as e:
                    st.error(f"Error cargando {filename}: {e}")
    return marcadores

def transformar_marcadores(client, frase, marcador_id, signo, marcadores):
    marcador = marcadores[marcador_id]
    polaridad = "ALTA PRESENCIA SOCIAL (+)" if signo == "+" else "BAJA PRESENCIA SOCIAL (-)"
    regla_especifica = marcador["pos"] if signo == "+" else marcador["neg"]

    ejemplos = {
        "1": {
            "+": "Ejemplo: 'Los datos avanzan con cierta terquedad, como si se resistieran a encajar del todo en la hipótesis inicial.'",
            "-": "Ejemplo: 'El procedimiento se ejecuta correctamente conforme a los parámetros establecidos.'"
        },
        "2": {
            "+": "Ejemplo: 'Podríamos considerar esta conclusión, aunque quizá existan otras lecturas posibles.'",
            "-": "Ejemplo: 'La interpretación correcta ha sido establecida.'"
        },
        "3": {
            "+": "Ejemplo: 'Después de revisarlo, considero que estamos dejando fuera una variable importante.'",
            "-": "Ejemplo: 'Se recomienda revisar la información.'"
        },
        "4": {
            "+": "Ejemplo: '¡Qué hallazgo tan interesante! Realmente abre nuevas posibilidades.'",
            "-": "Ejemplo: 'El hallazgo presenta relevancia analítica.'"
        }
    }

    ejemplo_contexto = ejemplos.get(marcador_id, {}).get(signo, "")

    system_instruction = (
        "ERES UN EXPERTO EN PSICOLINGÜÍSTICA Y PROCESAMIENTO DE LENGUAJE NATURAL EN ESPAÑOL.\n"
        f"TU TAREA ES REESCRIBIR LA FRASE PROPORCIONADA APLICANDO EL MARCADOR: '{marcador['nombre']}'.\n\n"
        f"CONTEXTO TÉCNICO:\n"
        f"- Marcador: {marcador['nombre']}\n"
        f"- Definición: {marcador['def']}\n"
        f"- Objetivo: {polaridad}\n"
        f"- Reglas de Aplicación: {regla_especifica}\n"
        f"- {ejemplo_contexto}\n\n"
        "REGLAS CRÍTICAS DE SALIDA:\n"
        "1. Devuelve ÚNICAMENTE la frase transformada.\n"
        "2. No incluyas explicaciones, etiquetas, comillas ni preámbulos.\n"
        "3. Mantén el significado central pero altera profundamente el estilo lingüístico.\n"
        "4. GRAMÁTICA: Asegura el uso correcto de signos de apertura (¿, ¡) y cierre (?, !) según las normas del Español.\n"
        "5. Maximiza el uso de los indicadores mencionados en las Reglas de Aplicación."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"FRASE ORIGINAL: {frase}"}
            ],
            model=DEFAULT_MODEL,
            temperature=0.4,
            max_tokens=150,
            top_p=0.9
        )
        return chat_completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        return f"ERROR DE PROCESAMIENTO: {str(e)}"

def transformar_robotico(client, concepto):
    system_instruction = (
        "NODO_PROCESAMIENTO: RAW_CORE_3.2\n"
        "STATUS: PURE_TRANSFORMER\n\n"
        "REGLAS:\n"
        "1. TAREA: Convertir INPUT en una frase descriptiva técnica y plana.\n"
        "2. ESTRUCTURA: [Sujeto Inanimado] + [Verbo en Presente] + [Complemento Descriptivo].\n"
        "3. PROHIBICIONES: No usar 'Yo', 'Tú', 'Nosotros', ni saludos.\n"
        "4. ESTILO: Neutralidad absoluta. Sin adjetivos emocionales. Sin números.\n"
        "5. SALIDA: ÚNICAMENTE la frase transformada terminada en punto.\n\n"
        "EJEMPLO:\n"
        "Input: Clima\n"
        "Output: El fenómeno atmosférico manifiesta variaciones en la temperatura local."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"TRANSFORMAR_CONTENIDO: {concepto}"}
            ],
            model=DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=100,
            top_p=0.9
        )
        res = chat_completion.choices[0].message.content.strip().split('\n')[0]
        if not res.endswith('.'):
            res += '.'
        return res
    except Exception as e:
        return f"SYSTEM_FAILURE: {str(e)}"

# --- SIDEBAR (NAVEGACIÓN) ---
with st.sidebar:
    st.title("⚙️ Configuración")
    modo = st.selectbox("Selecciona Agente:", ["Marcadores Lingüísticos", "Nodo Robótico (Bruto)"])
    st.info("Este sistema utiliza Groq Cloud para procesamiento en tiempo real.")

# --- CUERPO PRINCIPAL ---
st.title("🚀 Suite de Inteligencia Lingüística")

client = get_groq_client()

if modo == "Marcadores Lingüísticos":
    st.header("🔡 Agente de Marcadores")
    st.write("Transforma frases aplicando variaciones de presencia social.")
    
    marcadores = cargar_marcadores()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        frase = st.text_area("Ingresa la frase:", placeholder="Ej: La reunión fue un éxito.")
    
    with col2:
        m_nombre = st.selectbox("Marcador:", [m["nombre"] for m in marcadores.values()])
        signo = st.radio("Signo:", ["+", "-"], help="+ para alta presencia social, - para baja.")
    
    if st.button("Transformar Frase"):
        if frase:
            m_id = [k for k, v in marcadores.items() if v["nombre"] == m_nombre][0]
            with st.spinner("Procesando transformación neural..."):
                resultado = transformar_marcadores(client, frase, m_id, signo, marcadores)
                st.subheader("Resultado:")
                st.markdown(f'<div class="result-box">{resultado}</div>', unsafe_allow_html=True)
        else:
            st.warning("Por favor, ingresa una frase.")

else:
    st.header("🤖 Nodo de Procesamiento Bruto")
    st.write("Elimina toda traza de heurística social y subjetividad.")
    
    concepto = st.text_input("Ingresa concepto o frase:", placeholder="Ej: Amistad")
    
    if st.button("Ejecutar Nodo RAW"):
        if concepto:
            with st.spinner("Bypassing Social Heuristics..."):
                resultado = transformar_robotico(client, concepto)
                st.subheader("Salida de Datos:")
                st.markdown(f'<div class="result-box">{resultado}</div>', unsafe_allow_html=True)
        else:
            st.warning("Ingresa un concepto para procesar.")

st.divider()
st.caption("Desarrollado para despliegue autónomo en la nube.")
