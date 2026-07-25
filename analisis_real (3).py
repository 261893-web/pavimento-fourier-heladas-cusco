"""
=======================================================================
 Modelo de series de Fourier para el comportamiento térmico de
 pavimentos rígidos bajo helada altoandina (San Jerónimo, Cusco)
 -----------------------------------------------------------------
 VERSIÓN FINAL VERIFICADA
 Fuente de datos: NASA POWER (MERRA-2), 2 años horarios (2024-2026)
=======================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y FÍSICOS (declarados y justificados en el paper, Sección 5.5)
# =======================================================================
L_LOSA = 4.5            # Longitud de losa típica entre juntas (m)
B_LOSA = 3.6             # Ancho de carril típico (m)
H_LOSA = 0.20            # Espesor de losa (m)
K_SUBRASANTE = 50.0e6    # Módulo de reacción de subrasante típico (Pa/m)

MR_DISENO_KGCM2 = 42                       # Módulo de Rotura mínimo, norma peruana
MR_PSI = MR_DISENO_KGCM2 * 14.223
E_CONCRETO = 5700 * MR_PSI * 0.00689476    # AASHTO: Ec = 5700*MR(psi) -> MPa
NU_CONCRETO = 0.15
ALPHA_C_MIN = 1.0e-5     # coef. expansión térmica, cota inferior (agregado andesítico)
ALPHA_C_MAX = 1.3e-5     # coef. expansión térmica, cota superior (agregado basáltico)
ALPHA_TERMICO = 0.0025   # difusividad térmica del concreto (m^2/h)

# =======================================================================
# 2. CARGA Y LIMPIEZA DE DATOS REALES (NASA POWER) — separador correcto: COMA
# =======================================================================
temperaturas, meses = [], []
with open("datos_kayra.csv") as f:
    lines = f.readlines()

start_idx = 0
for i, line in enumerate(lines):
    if line.startswith("YEAR"):
        start_idx = i + 1
        break

for line in lines[start_idx:]:
    parts = line.strip().split(",")     # <- separador correcto (el archivo es CSV real)
    if len(parts) != 5:
        continue
    year, mo, dy, hr, t2m = parts
    temperaturas.append(float(t2m))
    meses.append(int(mo))

temperaturas = np.array(temperaturas)
meses = np.array(meses)
t_horas = np.arange(len(temperaturas))

faltantes = (temperaturas == -999)
n_faltantes = faltantes.sum()
if n_faltantes > 0:
    idx_validos = np.where(~faltantes)[0]
    temperaturas[faltantes] = np.interp(np.where(faltantes)[0], idx_validos, temperaturas[idx_validos])

print(f"Registros horarios cargados: {len(temperaturas)}")
print(f"Datos faltantes interpolados: {n_faltantes} ({100*n_faltantes/len(temperaturas):.2f}%)")

# =======================================================================
# 3. FILTRADO A ÉPOCA SECA (mayo-agosto) — temporada crítica de helada
# =======================================================================
mask_seca = np.isin(meses, [5, 6, 7, 8])
temp_seca = temperaturas[mask_seca]
t_seca = np.arange(len(temp_seca))
print(f"Horas en época seca (may-ago): {len(temp_seca)}")

# =======================================================================
# 4. AJUSTE DE FOURIER (5 armónicos) SOBRE LA ÉPOCA SECA
# =======================================================================
def ajustar_fourier(t_eje, datos, n_armonicos=5, periodo=24):
    T_media = np.mean(datos)
    senal = datos - T_media
    omega = 2*np.pi/periodo
    a_n, b_n = [], []
    for n in range(1, n_armonicos+1):
        a_n.append(2/len(datos)*np.sum(senal*np.cos(n*omega*t_eje)))
        b_n.append(2/len(datos)*np.sum(senal*np.sin(n*omega*t_eje)))
    return T_media, np.array(a_n), np.array(b_n), omega

T_media, a_n, b_n, omega = ajustar_fourier(t_seca, temp_seca)
amplitud_1 = np.sqrt(a_n[0]**2 + b_n[0]**2)

recon = np.full_like(temp_seca, T_media, dtype=float)
for n,(a,b) in enumerate(zip(a_n,b_n), start=1):
    recon += a*np.cos(n*omega*t_seca) + b*np.sin(n*omega*t_seca)
rmse = np.sqrt(np.mean((temp_seca-recon)**2))
r2 = 1 - np.sum((temp_seca-recon)**2)/np.sum((temp_seca-np.mean(temp_seca))**2)

print(f"\nAmplitud armónico principal (época seca): {amplitud_1:.2f} °C")
print(f"RMSE del ajuste: {rmse:.3f} °C   |   R²: {r2:.3f}")

# =======================================================================
# 5. TEMPERATURA EN PROFUNDIDAD Y GRÁFICO
# =======================================================================
def temperatura_en_profundidad(x, t, T_media, a_n, b_n, omega, alpha):
    T = np.full_like(t, T_media, dtype=float)
    for n,(a,b) in enumerate(zip(a_n,b_n), start=1):
        d_n = np.sqrt(2*alpha/(n*omega))
        fase = n*omega*t - x/d_n
        T += np.exp(-x/d_n)*(a*np.cos(fase)+b*np.sin(fase))
    return T

ventana = slice(0, 24*5)
plt.figure(figsize=(10,5))
plt.plot(t_seca[ventana], temp_seca[ventana], "k--", label="Temperatura real (NASA POWER, época seca)", alpha=0.6)
for x in [0.0, 0.05, 0.10, 0.20]:
    T_x = temperatura_en_profundidad(x, t_seca[ventana], T_media, a_n, b_n, omega, ALPHA_TERMICO)
    plt.plot(t_seca[ventana], T_x, label=f"x = {x*100:.0f} cm")
plt.xlabel("Tiempo (horas, época seca)"); plt.ylabel("Temperatura (°C)")
plt.title("Distribución de temperatura en la losa — época seca (San Jerónimo, Cusco)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("resultado_real_temperatura.png", dpi=150)
print("\nGráfico guardado: resultado_real_temperatura.png")

# =======================================================================
# 6. COEFICIENTE DE BRADBURY (C) — método de tabla clásica verificada (IRC:58 / Bradbury 1938)
# =======================================================================
l_rigidez = (E_CONCRETO*1e6 * H_LOSA**3 / (12*(1-NU_CONCRETO**2)*K_SUBRASANTE))**0.25

tabla_bradbury = {1:0.000,2:0.040,3:0.175,4:0.440,5:0.720,6:0.920,7:1.030,
                  8:1.077,9:1.080,10:1.075,11:1.050,12:1.000}
def C_bradbury(ratio):
    lo = max(1, min(int(np.floor(ratio)), 12)); hi = max(1, min(lo+1, 12))
    if lo == hi: return tabla_bradbury[lo]
    return tabla_bradbury[lo] + (ratio-lo)*(tabla_bradbury[hi]-tabla_bradbury[lo])

L_l = L_LOSA/l_rigidez
B_l = B_LOSA/l_rigidez
C = max(C_bradbury(L_l), C_bradbury(B_l))

print(f"\nRadio de rigidez relativa l: {l_rigidez:.4f} m")
print(f"L/l = {L_l:.2f}  |  B/l = {B_l:.2f}  |  C (Bradbury) = {C:.3f}")

# =======================================================================
# 7. GRADIENTE Y ESFUERZO TÉRMICO (ciclo característico, época seca)
# =======================================================================
t_ciclo = np.arange(24)
T_sup = temperatura_en_profundidad(0.0, t_ciclo, T_media, a_n, b_n, omega, ALPHA_TERMICO)
T_20cm = temperatura_en_profundidad(0.20, t_ciclo, T_media, a_n, b_n, omega, ALPHA_TERMICO)
delta_T = (T_sup - T_20cm).max()

def sigma_bradbury(dT, E, alpha_c, nu, C):
    return (E*alpha_c*dT)/(2*(1-nu))*C

sigma_min = sigma_bradbury(delta_T, E_CONCRETO, ALPHA_C_MIN, NU_CONCRETO, C)
sigma_max = sigma_bradbury(delta_T, E_CONCRETO, ALPHA_C_MAX, NU_CONCRETO, C)

print(f"\nGradiente térmico máximo (época seca): {delta_T:.2f} °C")
print(f"E (AASHTO, via MR): {E_CONCRETO:.0f} MPa")
print(f"Esfuerzo de alabeo térmico: {sigma_min:.3f} - {sigma_max:.3f} MPa "
      f"(según αc entre {ALPHA_C_MIN:.1e} y {ALPHA_C_MAX:.1e} /°C)")

# =======================================================================
# 8. SEVERIDAD REAL (datos crudos, sin modelar) — contexto anual completo
# =======================================================================
n_dias = len(temperaturas)//24
amp_diaria = np.array([temperaturas[d*24:(d+1)*24].max()-temperaturas[d*24:(d+1)*24].min()
                        for d in range(n_dias)])
print(f"\n[Contexto anual completo] Amplitud diaria real: "
      f"promedio {amp_diaria.mean():.2f} °C, máxima {amp_diaria.max():.2f} °C")
print(f"[Contexto anual completo] Temperatura mínima absoluta: {temperaturas.min():.1f} °C")
