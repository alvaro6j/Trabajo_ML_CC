import pickle
import pandas as pd

# Cargar el modelo .pkl
with open('modelo_HistGradientBoosting.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Lista de características en el orden esperado por el modelo
columnas = ['Edad', 'Nivel_Educacional', 'Años_Trabajando', 'Ingresos', 'Deuda_Comercial', 'Deuda_Credito', 'Otras_Deudas', 'Ratio_Ingresos_Deudas']

# Crear una lista para almacenar los valores ingresados por el usuario
valores_usuario = []

# Valores permitidos para la variable categórica
niveles_validos = ["SupInc", "Med", "Bas", "Posg", "SupCom"]

print("Por favor, ingrese los siguientes valores:")
print("(Nota: Ingrese los valores correspondientes a cada característica.)\n")

for columna in columnas:
    if columna == 'Nivel_Educacional':
        while True:
            valor = input(f"{columna} (opciones válidas: {', '.join(niveles_validos)}): ").strip()
            if valor in niveles_validos:
                valores_usuario.append(valor)
                break
            else:
                print(f"Valor inválido. Debes ingresar uno de: {', '.join(niveles_validos)}")      
    else:
        while True:
            try:
                valor = float(input(f"{columna}: "))
                valores_usuario.append(valor)
                break
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número válido.")

# Convertir los valores a un DataFrame con las columnas correctas
nueva_muestra = pd.DataFrame([valores_usuario], columns=columnas)

# Realizar la predicción con el modelo cargado
prediccion = modelo.predict(nueva_muestra)
probabilidad = modelo.predict_proba(nueva_muestra)[:, 1]

# Mostrar el resultado al usuario
print("\n--- RESULTADO ---")
if prediccion[0] == 1:
    print("\n*** Rechazar crédito ***")
else:
    print("\nAceptar crédito")

print(f"Probabilidad de no pago: {probabilidad[0]:.4f}")