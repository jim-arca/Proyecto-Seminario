import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from st_audiorec import st_audiorec
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Analizador de Espectro de Audio",
    page_icon="🎙️",
    layout="wide"
)

# --- Estilos CSS Personalizados (Opcional) ---
st.markdown("""
<style>
    /* Estilos para que la app se vea más moderna */
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #4B8BBE;
        background-color: #4B8BBE;
        color: white;
    }
    .stButton>button:hover {
        border: 1px solid #3A6A94;
        background-color: #3A6A94;
    }
    h1 {
        color: #3A6A94;
    }
</style>
""", unsafe_allow_html=True)


# --- Título de la Aplicación ---
st.title("🎙️ Analizador de Espectro de Audio")
st.write(
    "Graba audio desde tu micrófono y visualiza su espectrograma en tiempo real. "
    "Presiona el ícono del micrófono para comenzar a grabar."
)

# --- Widget para Grabar Audio ---
# Este componente gestiona el acceso al micrófono desde el navegador.
wav_audio_data = st_audiorec()

# Espacio en blanco para mejorar el diseño
st.write("")
st.write("---")

# --- Panel de Configuración ---
st.header("⚙️ Ingrese a la configuración")

# Dividimos el área de configuración en columnas para un mejor diseño
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Selector para la escala de frecuencia
    escala_frecuencia = st.selectbox(
        "Escala de frecuencia",
        ("Lineal", "Logarítmica"),
        index=1 # Por defecto, logarítmica
    )

with col2:
    # Selector para el mapa de colores del espectrograma
    mapa_color = st.selectbox(
        "Color",
        ("Rojo", "Azul", "Verde", "Plasma", "Inferno", "Viridis"),
        index=0 # Por defecto, Rojo
    )
    # Mapeo de nombres de color a mapas de color de Matplotlib
    color_map_dict = {
        "Rojo": "Reds",
        "Azul": "Blues",
        "Verde": "Greens",
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Viridis": "viridis"
    }
    selected_cmap = color_map_dict[mapa_color]

with col3:
    # Selector para el brillo o estilo del tema
    brillo = st.selectbox(
        "Brillo",
        ("Normal", "Oscuro"),
        index=0
    )
    plot_face_color = 'white' if brillo == 'Normal' else '#0E1117'
    plot_text_color = 'black' if brillo == 'Normal' else 'white'


# --- Procesamiento y Visualización del Audio ---
if wav_audio_data is not None:
    st.subheader("Análisis del Audio Grabado")
    try:
        # Convertimos los datos de audio en bytes a un formato que podamos usar
        audio_bytes = io.BytesIO(wav_audio_data)
        
        # Leemos el archivo WAV virtual
        samplerate, data = wavfile.read(audio_bytes)
        
        # Si el audio es estéreo, tomamos solo un canal
        if len(data.shape) > 1:
            data = data[:, 0]
            
        st.write(f"**Tasa de muestreo:** `{samplerate} Hz`")
        st.write(f"**Duración:** `{len(data)/samplerate:.2f} segundos`")

        # Mostramos el reproductor de audio
        st.audio(wav_audio_data, format='audio/wav')

        # --- Generación del Espectrograma ---
        st.write("### Espectrograma de Frecuencia")

        # Creamos la figura para el gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Establecemos el color de fondo y de texto según la selección
        fig.patch.set_facecolor(plot_face_color)
        ax.set_facecolor(plot_face_color)
        
        # Calculamos el espectrograma
        frecuencias, tiempos, Sxx = signal.spectrogram(data, samplerate)

        # Graficamos el espectrograma
        mesh = ax.pcolormesh(tiempos, frecuencias, 10 * np.log10(Sxx + 1e-9), cmap=selected_cmap, shading='gouraud')
        
        # Configuramos etiquetas y título con el color adecuado
        ax.set_ylabel('Frecuencia [Hz]', color=plot_text_color)
        ax.set_xlabel('Tiempo [s]', color=plot_text_color)
        ax.set_title('Espectrograma', color=plot_text_color)

        # Configuramos los colores de los ejes
        ax.tick_params(axis='x', colors=plot_text_color)
        ax.tick_params(axis='y', colors=plot_text_color)
        ax.spines['bottom'].set_color(plot_text_color)
        ax.spines['top'].set_color(plot_text_color)
        ax.spines['left'].set_color(plot_text_color)
        ax.spines['right'].set_color(plot_text_color)
        
        # Aplicamos la escala de frecuencia seleccionada
        if escala_frecuencia == 'Logarítmica':
            ax.set_yscale('log')
            # Evitar que el eje Y llegue a cero en escala logarítmica
            ax.set_ylim(bottom=max(1, frecuencias[frecuencias > 0].min()), top=samplerate / 2)
            
        # Añadimos una barra de color
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Intensidad [dB]', color=plot_text_color)
        cbar.ax.tick_params(colors=plot_text_color)

        # Mostramos el gráfico en Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el audio: {e}")
        st.info("Asegúrate de que la grabación contenga audio y no sea demasiado corta.")
else:
    st.info("Esperando a que se grabe un audio...")
