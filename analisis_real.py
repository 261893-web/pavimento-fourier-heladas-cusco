import numpy as np
import matplotlib.pyplot as plt
import csv

# =======================================================================
# 1. CARGAR DATOS REALES DE NASA POWER (San Jerónimo / Kayra)
# =======================================================================
horas_totales = []
temperaturas = []

with open("datos_kayra.csv") as f:
    lines = f.readlines()

# saltar el header hasta encontrar la línea de columnas
start_idx = 0
for i, line in enumerate(lines):
    if line.startswith("YEAR"):
        start_idx = i + 1
        break

for line in lines[start_idx:]:
    parts = line.strip().split(",")
    if len(parts) < 5:
        continue
    year, mo, dy, hr, t2m = parts
    temperaturas.append(float(t2m))

temperaturas = np.array(temperaturas)
t_horas = np.arange(len(temperaturas))  # 0,1,2,... horas consecutivas

# --- Limpieza: NASA POWER marca los datos faltantes con -999 ---
faltantes = (temperaturas == -999)
n_faltantes = faltantes.sum()
if n_faltantes > 0:
    # interpolación lineal simple sobre los huecos
    indices_validos = np.where(~faltantes)[0]
    temperaturas[faltantes] = np.interp(
        np.where(faltantes)[0], indices_validos, temperaturas[indices_validos]
    )
    print(f"Se interpolaron {n_faltantes} registros faltantes (-999), "
          f"equivalentes al {100*n_faltantes/len(temperaturas):.2f}% del total.")

print(f"Total de registros horarios cargados: {len(temperaturas)}")
print(f"Temperatura mínima registrada: {temperaturas.min():.1f} °C")
print(f"Temperatura máxima registrada: {temperaturas.max():.1f} °C")
print(f"Temperatura media: {temperaturas.mean():.2f} °C")

# =======================================================================
# 2. AJUSTE DE FOURIER (5 armónicos, sobre el ciclo diario de 24h)
# =======================================================================
def ajustar_fourier(t_horas, temperaturas, n_armonicos=5, periodo_horas=24):
    N = len(temperaturas)
    T_media = np.mean(temperaturas)
    señal = temperaturas - T_media
    omega = 2 * np.pi / periodo_horas
    a_n, b_n = [], []
    for n in range(1, n_armonicos + 1):
        a = 2 / N * np.sum(señal * np.cos(n * omega * t_horas))
        b = 2 / N * np.sum(señal * np.sin(n * omega * t_horas))
        a_n.append(a)
        b_n.append(b)
    return T_media, np.array(a_n), np.array(b_n), omega

T_media, a_n, b_n, omega = ajustar_fourier(t_horas, temperaturas, n_armonicos=5)
print("\n--- Coeficientes de Fourier (ciclo diario promedio) ---")
print("Temperatura media:", round(T_media, 2), "°C")
print("Coeficientes a_n:", np.round(a_n, 3))
print("Coeficientes b_n:", np.round(b_n, 3))

# amplitud del primer armónico (oscilación diaria dominante)
amplitud_1 = np.sqrt(a_n[0]**2 + b_n[0]**2)
print(f"\nAmplitud de la oscilación diaria principal: {amplitud_1:.2f} °C")

# =======================================================================
# 3. TEMPERATURA A DISTINTAS PROFUNDIDADES
# =======================================================================
def temperatura_en_profundidad(x, t, T_media, a_n, b_n, omega, alpha):
    T = np.full_like(t, T_media, dtype=float)
    for n, (a, b) in enumerate(zip(a_n, b_n), start=1):
        d_n = np.sqrt(2 * alpha / (n * omega))
        fase = n * omega * t - x / d_n
        T += np.exp(-x / d_n) * (a * np.cos(fase) + b * np.sin(fase))
    return T

alpha_concreto = 0.0025  # m^2/h (referencia de literatura, pendiente de ajuste fino)
profundidades = [0.0, 0.05, 0.10, 0.20]

# graficamos solo los primeros 5 días para que se vea claro
ventana = slice(0, 24*5)

plt.figure(figsize=(10, 5))
plt.plot(t_horas[ventana], temperaturas[ventana], "k--", label="Temperatura real (NASA POWER)", alpha=0.6)
for x in profundidades:
    T_x = temperatura_en_profundidad(x, t_horas[ventana], T_media, a_n, b_n, omega, alpha_concreto)
    plt.plot(t_horas[ventana], T_x, label=f"x = {x*100:.0f} cm")
plt.xlabel("Tiempo (horas)")
plt.ylabel("Temperatura (°C)")
plt.title("Distribución de temperatura en la losa — datos reales (San Jerónimo, Cusco)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("resultado_real_temperatura.png", dpi=150)
print("\nGráfico guardado: resultado_real_temperatura.png")

# =======================================================================
# 4. ESFUERZO TÉRMICO (Bradbury) CON VALORES REALES DEL PERÚ
# =======================================================================
def esfuerzo_termico_bradbury(delta_T, E, alpha_c, nu, C=1.0):
    return (E * alpha_c * delta_T) / (2 * (1 - nu)) * C

# usamos el gradiente diario típico (superficie vs 20cm) del PRIMER día real
T_superficie = temperatura_en_profundidad(0.0, t_horas[:24], T_media, a_n, b_n, omega, alpha_concreto)
T_20cm = temperatura_en_profundidad(0.20, t_horas[:24], T_media, a_n, b_n, omega, alpha_concreto)
delta_T_dia1 = (T_superficie - T_20cm).max()

sigma = esfuerzo_termico_bradbury(delta_T_dia1, E=24600, alpha_c=1.3e-5, nu=0.15)
print(f"\nGradiente térmico máximo (superficie vs 20cm), día de ejemplo: {delta_T_dia1:.2f} °C")
print(f"Esfuerzo térmico estimado (Bradbury): {sigma:.3f} MPa")

# Resumen general de los 2 años
print(f"\n--- RESUMEN GENERAL (2 años de datos reales) ---")
print(f"Amplitud térmica diaria promedio: {amplitud_1:.2f} °C")
print(f"Rango total registrado: {temperaturas.min():.1f} °C a {temperaturas.max():.1f} °C")
