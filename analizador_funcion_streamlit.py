# analizador_funcion_streamlit.py
# Streamlit app que permite introducir una función de x, calcula puntos críticos,
# intervalos de crecimiento/decrecimiento/constancia, máximos y mínimos locales,
# y muestra la gráfica con marcas.

import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

st.set_page_config(page_title="Analizador de funciones", layout="wide")
st.title("Analizador de funciones — Puntos críticos, monotonicidad y gráfica - Hecho por: Angela Alfaro - Yilmer Diaz - Erika Molina")

# Entrada
func_text = st.text_input("Ingrese la función en variable x")
col1, col2 = st.columns(2)
with col1:
    x_min = st.number_input("Dominio: x mínimo", value=-10.0, step=1.0)
    x_max = st.number_input("Dominio: x máximo", value=10.0, step=1.0)
with col2:
    samples = st.slider("Puntos muestreo para búsqueda numérica", 200, 2000, 800)
    tol = st.number_input("Tolerancia numérica (derivada ≈ 0)", value=1e-6, format="%.0e")

if x_min >= x_max:
    st.error("Asegúrese de que x mínimo < x máximo")
    st.stop()

x = sp.symbols('x')

# Intentar parsear la función
try:
    f_sym = sp.sympify(func_text)
except Exception as e:
    st.error(f"No pude interpretar la función: {e}")
    st.stop()

# Derivadas simbólicas
f1_sym = sp.diff(f_sym, x)
f2_sym = sp.diff(f1_sym, x)

# Lambdify para evaluaciones numéricas
f_num = sp.lambdify(x, f_sym, modules=["numpy"])
f1_num = sp.lambdify(x, f1_sym, modules=["numpy"])
f2_num = sp.lambdify(x, f2_sym, modules=["numpy"]) 

# Búsqueda numérica de raíces de la derivada en el intervalo
xs = np.linspace(x_min, x_max, samples)
der_vals = None
try:
    der_vals = f1_num(xs)
except Exception:
    # Algunas funciones (ej: con log) pueden lanzar errores al evaluar fuera de dominio
    der_vals = np.array([np.nan if np.isnan(xi) else f1_num(xi) for xi in xs])

candidates = set()
from sympy import nsolve

# Usar cambios de signo en la derivada para inicializar nsolve
for i in range(len(xs)-1):
    a, b = xs[i], xs[i+1]
    da, db = der_vals[i], der_vals[i+1]
    if np.isnan(da) or np.isnan(db):
        continue
    if da == 0:
        candidates.add(a)
    if da*db < 0:
        try:
            root = float(sp.nsolve(f1_sym, x, (a+b)/2))
            if x_min - 1e-8 <= root <= x_max + 1e-8:
                candidates.add(root)
        except Exception:
            # nsolve puede fallar; en ese caso aproximamos con bisección
            try:
                # bisección simple por numpy
                rr = np.nan
                for _ in range(40):
                    m = (a+b)/2
                    if f1_num(m) == 0:
                        rr = m
                        break
                    if np.sign(f1_num(a)) * np.sign(f1_num(m)) < 0:
                        b = m
                    else:
                        a = m
                if not np.isnan(rr):
                    candidates.add(rr)
            except Exception:
                pass

# convertir a lista ordenada y eliminar duplicados cercanos
crit_points = sorted(list(candidates))
# incluir puntos donde la derivada es indefinida (denominador cero) - intentar detectar
try:
    num, den = sp.fraction(sp.simplify(sp.together(f1_sym)))
    den_zero = sp.solve(sp.Eq(den, 0), x)
    for z in den_zero:
        try:
            zv = float(z)
            if x_min <= zv <= x_max:
                crit_points.append(zv)
        except Exception:
            pass
except Exception:
    pass

# filtrar y ordenar
crit_points = sorted(list(set([round(float(c), 12) for c in crit_points if np.isfinite(c)])))

# Agregar límites del dominio para analizar intervalos
breaks = [x_min] + crit_points + [x_max]
intervals = []
classification = []
for i in range(len(breaks)-1):
    a = breaks[i]
    b = breaks[i+1]
    mid = (a+b)/2
    try:
        dmid = float(f1_num(mid))
    except Exception:
        dmid = np.nan
    if np.isnan(dmid):
        cls = 'indeterminado'
    elif abs(dmid) <= tol:
        cls = 'constante'
    elif dmid > 0:
        cls = 'creciente'
    else:
        cls = 'decreciente'
    intervals.append((a,b))
    classification.append(cls)

# Clasificar máximos y mínimos
extrema = []
for c in crit_points:
    # evaluar segunda derivada cuando posible
    try:
        s = float(f2_num(c))
        if s > 0:
            tipo = 'mínimo local (segunda derivada > 0)'
        elif s < 0:
            tipo = 'máximo local (segunda derivada < 0)'
        else:
            # fallback a cambio de signo
            # mirar pequeños pasos a la izquierda y derecha
            h = 1e-4
            try:
                left = f1_num(c - h)
                right = f1_num(c + h)
                if left > 0 and right < 0:
                    tipo = 'máximo local (cambio + -> -)'
                elif left < 0 and right > 0:
                    tipo = 'mínimo local (cambio - -> +)'
                else:
                    tipo = 'punto crítico (no concluyente)'
            except Exception:
                tipo = 'punto crítico (no concluyente)'
    except Exception:
        tipo = 'punto crítico (segunda derivada no evaluable)'
    # valor de la función
    try:
        fv = float(f_num(c))
    except Exception:
        fv = None
    extrema.append((c, fv, tipo))

# Mostrar resultados
st.header('Resultados simbólicos')
st.markdown(f"**Función:** ${sp.latex(f_sym)}$")
st.markdown(f"**Primera derivada:** ${sp.latex(sp.simplify(f1_sym))}$")
st.markdown(f"**Segunda derivada:** ${sp.latex(sp.simplify(f2_sym))}$")

st.subheader('Puntos críticos encontrados (en el intervalo)')
if crit_points:
    for c in crit_points:
        st.write(f"x = {c}")
else:
    st.write("No se encontraron puntos críticos en el intervalo dado.")

st.subheader('Intervalos y clasificación')
for (a,b), cls in zip(intervals, classification):
    st.write(f"({a}, {b}): {cls}")

st.subheader('Extremos locales (clasificación)')
if extrema:
    for c, fv, tipo in extrema:
        st.write(f"x = {c}    f(x) = {fv}    -> {tipo}")
else:
    st.write('No hay extremos locales encontrados.')

# Gráfica
st.header('Gráfica')
plot_x = np.linspace(x_min, x_max, 1000)
with st.expander('Mostrar gráfica (matplotlib)'):
    fig, ax = plt.subplots(figsize=(8,4))
    try:
        y = f_num(plot_x)
        ax.plot(plot_x, y)
    except Exception:
        # evaluar punto a punto evitando NaN
        y = np.array([np.nan]*len(plot_x))
        for i, xv in enumerate(plot_x):
            try:
                y[i] = f_num(xv)
            except Exception:
                y[i] = np.nan
        ax.plot(plot_x, y)

    # marcar puntos críticos
    for c, fv, tipo in extrema:
        if fv is not None:
            if 'mínimo' in tipo:
                ax.scatter(c, fv, marker='o', s=60)
                ax.annotate('min', xy=(c,fv), xytext=(5,5), textcoords='offset points')
            elif 'máximo' in tipo:
                ax.scatter(c, fv, marker='^', s=60)
                ax.annotate('max', xy=(c,fv), xytext=(5,5), textcoords='offset points')
            else:
                ax.scatter(c, fv, marker='x', s=50)
                ax.annotate('crit', xy=(c,fv), xytext=(5,5), textcoords='offset points')

    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Gráfica de la función y puntos críticos')
    ax.xaxis.set_major_locator(MaxNLocator(10))
    st.pyplot(fig)

st.write('---')
