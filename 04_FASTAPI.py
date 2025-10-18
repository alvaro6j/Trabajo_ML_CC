from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el scaler desde los archivos .pkl

with open('modelo_HistGradientBoosting.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Definir las características esperadas
columnas = ['Edad', 'Nivel_Educacional', 'Años_Trabajando', 'Ingresos', 'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas']

# Validar variable Nivel_Educacional
class NivelEducacionalEnum(str, Enum):
    Bas = "Bas"
    Med = "Med"
    SupInc = "SupInc"
    SupCom = "SupCom"
    Posg = "Posg"

# Crear la aplicación FastAPI
app = FastAPI(title="Detección incumplimiento de pago")

# Definir el modelo de datos de entrada utilizando Pydantic
class Transaccion(BaseModel):
    Edad: float = Field(..., gt=0, description="Edad del solicitante (en años)")
    Nivel_Educacional: NivelEducacionalEnum
    Años_Trabajando: float
    Ingresos: float
    Deuda_Comercial: float
    Deuda_Credito: float
    Otras_Deudas: float
    Ratio_Ingresos_Deudas: float
    
# Definir el endpoint para predicción
@app.post("/prediccion/")
async def predecir_incumplimiento(transaccion: Transaccion):
    try:
        # Convertir la entrada en un DataFrame
        datos_entrada = pd.DataFrame([transaccion.dict()], columns=columnas)

        # Predicción
        prediccion = modelo.predict(datos_entrada)
        probabilidad = modelo.predict_proba(datos_entrada)[:, 1]
               
        # Construir la respuesta
        resultado = {
            "NoPaga": bool(prediccion[0]),
            "Probabilidad_Incumplimiento": float(probabilidad[0])
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
