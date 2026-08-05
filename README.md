# RAG-Agentes

Este proyecto contiene un notebook de ejemplo para conectar un modelo de lenguaje con Google Gemini y usar herramientas externas de búsqueda web con Tavily.

## ¿Qué incluye?

El notebook `conectando-LLM.ipynb` muestra cómo:

- configurar la API de Google Gemini;
- listar modelos disponibles para generación de texto;
- crear un cliente de `ChatGoogleGenerativeAI`;
- construir una cadena simple con `PromptTemplate` y `StrOutputParser`;
- integrar una herramienta personalizada para buscar información en la web con Tavily.

## Requisitos

- Python 3.9 o superior
- Jupyter Notebook o JupyterLab
- acceso a una API key de Google Gemini
- acceso a una API key de Tavily

## Instalación de dependencias

En la primera celda del notebook se instalan las librerías necesarias:

```bash
pip install -U langchain
pip install -U langgraph
pip install -U google-generativeai
pip install -U langchain-google-genai
pip install -U python-dotenv
pip install -U langchain-community langchain-core tavily-python
```

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto con este contenido:

```env
GEMINI_API_KEY=tu_clave_de_gemini
TAVILY_API_KEY=tu_clave_de_tavily
```

## Cómo ejecutar

1. Abre el notebook `conectando-LLM.ipynb`.
2. Ejecuta las celdas en orden.
3. Asegúrate de haber cargado correctamente las variables desde `.env`.
4. Prueba la herramienta de búsqueda ejecutando la celda final que invoca `busca_web`.

## Notas

- Si el modelo no aparece en la lista, revisa que tu API key de Gemini sea válida.
- Si la búsqueda web falla, confirma que `TAVILY_API_KEY` esté configurada correctamente.
