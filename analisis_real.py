"""
=======================================================================
 Fourier-series model of the thermal behaviour of rigid pavements
 under high-Andean frost (San Jeronimo, Cusco, Peru)
 -----------------------------------------------------------------
 Data source: NASA POWER (MERRA-2), hourly, 23 Jul 2024 - 23 Jul 2026

 Usage:
     python analisis_real.py                 # uses the NASA POWER CSV in this folder
     python analisis_real.py path/to/file.csv

 Outputs (written next to this script):
     resultado_real_temperatura.png   temperature in the slab, 5 dry-season days
     console summary                   fit quality, gradient and curling stress
=======================================================================
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no display needed: works on servers, CI and any OS
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATTERN = "POWER_Point_Hourly_*.csv"
MISSING = -999.0

# =======================================================================
# 1. GEOMETRIC AND PHYSICAL PARAMETERS
# =======================================================================
L_LOSA = 4.5             # joint spacing (m)
B_LOSA = 3.6             # lane width (m)
H_LOSA = 0.20            # slab thickness (m)
K_SUBRASANTE = 50.0e6    # modulus of subgrade reaction (Pa/m)

MR_DISENO_KGCM2 = 42                       # minimum modulus of rupture, Peruvian standard
MR_PSI = MR_DISENO_KGCM2 * 14.223
E_CONCRETO = 5700 * MR_PSI * 0.00689476    # AASHTO: Ec = 5700*MR(psi), converted to MPa
NU_CONCRETO = 0.15
ALPHA_C_MIN = 1.0e-5     # thermal expansion coefficient, lower bound (1/degC)
ALPHA_C_MAX = 1.3e-5     # thermal expansion coefficient, upper bound (1/degC)
ALPHA_TERMICO = 0.0025   # thermal diffusivity of concrete (m^2/h)

N_ARMONICOS = 5
PERIODO_H = 24
MESES_SECA = [5, 6, 7, 8]   # dry season: May-August


# =======================================================================
# 2. DATA LOADING AND CLEANING
# =======================================================================
def encontrar_csv(argv):
    """Return the CSV path given on the command line, or the NASA POWER file next to the script."""
    if len(argv) > 1:
        ruta = argv[1]
        if not os.path.isfile(ruta):
            sys.exit(f"ERROR: file not found: {ruta}")
        return ruta
    candidatos = sorted(glob.glob(os.path.join(HERE, DATA_PATTERN)))
    if not candidatos:
        sys.exit(
            "ERROR: no NASA POWER file found.\n"
            f"Place a file matching '{DATA_PATTERN}' in {HERE}\n"
            "or run: python analisis_real.py path/to/file.csv"
        )
    return candidatos[0]


def leer_nasa_power(ruta):
    """Read the hourly NASA POWER CSV (header block + YEAR,MO,DY,HR,T2M columns)."""
    with open(ruta, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("YEAR"):
            start_idx = i + 1
            break
    if start_idx is None:
        sys.exit(f"ERROR: column header 'YEAR,MO,DY,HR,T2M' not found in {ruta}")

    temperaturas, meses, omitidas = [], [], 0
    for line in lines[start_idx:]:
        parts = line.strip().split(",")
        if len(parts) != 5:
            omitidas += int(bool(line.strip()))
            continue
        try:
            mo, t2m = int(parts[1]), float(parts[4])
        except ValueError:
            omitidas += 1
            continue
        temperaturas.append(t2m)
        meses.append(mo)

    if not temperaturas:
        sys.exit(f"ERROR: no valid data rows in {ruta}")
    if omitidas:
        print(f"Warning: {omitidas} malformed line(s) skipped")
    return np.array(temperaturas, dtype=float), np.array(meses)


def rellenar_faltantes(temperaturas):
    """Linearly interpolate missing values (-999). Values at the end of the record are held constant."""
    faltantes = np.isclose(temperaturas, MISSING)
    n = int(faltantes.sum())
    if n == len(temperaturas):
        sys.exit("ERROR: every value in the file is missing (-999)")
    if n:
        validos = np.where(~faltantes)[0]
        temperaturas = temperaturas.copy()
        temperaturas[faltantes] = np.interp(np.where(faltantes)[0], validos, temperaturas[validos])
    return temperaturas, n


# =======================================================================
# 3. MODEL
# =======================================================================
def ajustar_fourier(t_eje, datos, n_armonicos=N_ARMONICOS, periodo=PERIODO_H):
    T_media = np.mean(datos)
    senal = datos - T_media
    omega = 2 * np.pi / periodo
    a_n, b_n = [], []
    for n in range(1, n_armonicos + 1):
        a_n.append(2 / len(datos) * np.sum(senal * np.cos(n * omega * t_eje)))
        b_n.append(2 / len(datos) * np.sum(senal * np.sin(n * omega * t_eje)))
    return T_media, np.array(a_n), np.array(b_n), omega


def reconstruir(t_eje, T_media, a_n, b_n, omega):
    recon = np.full(len(t_eje), T_media, dtype=float)
    for n, (a, b) in enumerate(zip(a_n, b_n), start=1):
        recon += a * np.cos(n * omega * t_eje) + b * np.sin(n * omega * t_eje)
    return recon


def temperatura_en_profundidad(x, t, T_media, a_n, b_n, omega, alpha):
    """Periodic solution of the 1D heat equation (homogeneous half-space)."""
    T = np.full(len(t), T_media, dtype=float)
    for n, (a, b) in enumerate(zip(a_n, b_n), start=1):
        d_n = np.sqrt(2 * alpha / (n * omega))
        fase = n * omega * t - x / d_n
        T += np.exp(-x / d_n) * (a * np.cos(fase) + b * np.sin(fase))
    return T


TABLA_BRADBURY = {1: 0.000, 2: 0.040, 3: 0.175, 4: 0.440, 5: 0.720, 6: 0.920, 7: 1.030,
                  8: 1.077, 9: 1.080, 10: 1.075, 11: 1.050, 12: 1.000}


def C_bradbury(ratio):
    """Bradbury coefficient, linear interpolation of the classical chart."""
    lo = max(1, min(int(np.floor(ratio)), 12))
    hi = max(1, min(lo + 1, 12))
    if lo == hi:
        return TABLA_BRADBURY[lo]
    return TABLA_BRADBURY[lo] + (ratio - lo) * (TABLA_BRADBURY[hi] - TABLA_BRADBURY[lo])


def sigma_bradbury(dT, E, alpha_c, nu, C):
    return (E * alpha_c * dT) / (2 * (1 - nu)) * C


# =======================================================================
# 4. MAIN
# =======================================================================
def main(argv):
    ruta = encontrar_csv(argv)
    print(f"Data file: {os.path.basename(ruta)}")

    temperaturas, meses = leer_nasa_power(ruta)
    temperaturas, n_faltantes = rellenar_faltantes(temperaturas)
    print(f"Hourly records loaded: {len(temperaturas)}")
    print(f"Missing values interpolated: {n_faltantes} ({100 * n_faltantes / len(temperaturas):.2f}%)")

    # Dry season
    mask_seca = np.isin(meses, MESES_SECA)
    temp_seca = temperaturas[mask_seca]
    if len(temp_seca) < PERIODO_H:
        sys.exit("ERROR: not enough dry-season (May-August) data in the file")
    t_seca = np.arange(len(temp_seca))
    print(f"Dry-season hours (May-Aug): {len(temp_seca)}")

    # Fourier fit
    T_media, a_n, b_n, omega = ajustar_fourier(t_seca, temp_seca)
    amplitud_1 = np.sqrt(a_n[0] ** 2 + b_n[0] ** 2)
    recon = reconstruir(t_seca, T_media, a_n, b_n, omega)
    rmse = np.sqrt(np.mean((temp_seca - recon) ** 2))
    r2 = 1 - np.sum((temp_seca - recon) ** 2) / np.sum((temp_seca - np.mean(temp_seca)) ** 2)
    print(f"\nFirst-harmonic amplitude (dry season): {amplitud_1:.2f} C")
    print(f"Fit RMSE: {rmse:.3f} C   |   R2: {r2:.3f}")

    # Temperature at depth: figure
    ventana = slice(0, PERIODO_H * 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_seca[ventana], temp_seca[ventana], "k--", alpha=0.6,
            label="Measured air temperature (NASA POWER, dry season)")
    for x in [0.0, 0.05, 0.10, 0.20]:
        T_x = temperatura_en_profundidad(x, t_seca[ventana], T_media, a_n, b_n, omega, ALPHA_TERMICO)
        ax.plot(t_seca[ventana], T_x, label=f"x = {x * 100:.0f} cm")
    ax.set_xlabel("Time (hours, dry season)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature distribution in the slab, dry season (San Jerónimo, Cusco)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    salida_png = os.path.join(HERE, "resultado_real_temperatura.png")
    fig.savefig(salida_png, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {os.path.basename(salida_png)}")

    # Bradbury coefficient
    l_rigidez = (E_CONCRETO * 1e6 * H_LOSA ** 3 / (12 * (1 - NU_CONCRETO ** 2) * K_SUBRASANTE)) ** 0.25
    L_l, B_l = L_LOSA / l_rigidez, B_LOSA / l_rigidez
    C = max(C_bradbury(L_l), C_bradbury(B_l))
    print(f"\nRadius of relative stiffness l: {l_rigidez:.4f} m")
    print(f"L/l = {L_l:.2f}  |  B/l = {B_l:.2f}  |  C (Bradbury) = {C:.3f}")

    # Gradient and curling stress, characteristic cycle
    t_ciclo = np.arange(PERIODO_H)
    T_sup = temperatura_en_profundidad(0.0, t_ciclo, T_media, a_n, b_n, omega, ALPHA_TERMICO)
    T_fondo = temperatura_en_profundidad(H_LOSA, t_ciclo, T_media, a_n, b_n, omega, ALPHA_TERMICO)
    delta_T = (T_sup - T_fondo).max()
    sigma_min = sigma_bradbury(delta_T, E_CONCRETO, ALPHA_C_MIN, NU_CONCRETO, C)
    sigma_max = sigma_bradbury(delta_T, E_CONCRETO, ALPHA_C_MAX, NU_CONCRETO, C)
    print(f"\nMaximum thermal gradient (dry season): {delta_T:.2f} C")
    print(f"E (AASHTO, from MR): {E_CONCRETO:.0f} MPa")
    print(f"Curling stress: {sigma_min:.3f} - {sigma_max:.3f} MPa "
          f"(alpha_c between {ALPHA_C_MIN:.1e} and {ALPHA_C_MAX:.1e} 1/C)")

    # Raw daily severity, full record
    n_dias = len(temperaturas) // PERIODO_H
    dias = temperaturas[: n_dias * PERIODO_H].reshape(n_dias, PERIODO_H)
    amp_diaria = dias.max(axis=1) - dias.min(axis=1)
    print(f"\n[Full record] Raw daily range: mean {amp_diaria.mean():.2f} C, max {amp_diaria.max():.2f} C")
    print(f"[Full record] Absolute minimum temperature: {temperaturas.min():.1f} C")


if __name__ == "__main__":
    main(sys.argv)
