import base64
import io
import math
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm, shapiro

# App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title ="shadowy statesman"
server = app.server

app.layout = html.Div(
    id="main-container",
    className="modo-claro",
    children=[

        #         ENCABEZADO INSTITUCIONAL 
        html.Div(
            className="header",
            children=[
                html.Img(src="/assets/logo_unimag.png", className="logo", height='80px'),
                html.Div(
                    [
                        html.H1("UNIVERSIDAD DEL MAGDALENA", className="titulo-principal"),
                        html.H2("Facultad de Ingeniería Industrial – Control Estadístico de Procesos", className="subtitulo")
                    ],
                    className="titulo-container"
                ),
                html.Button("🌙", id="toggle-theme", n_clicks=0, className="theme-btn")
            ]
        ),

        # Línea dorada decorativa
        html.Div(className="gold-line"),

        #       CONTENIDO PRINCIPAL 
        html.Div(
            className="container",
            children=[

                # Columna izquierda
                html.Div(
                    className="sidebar",
                    children=[
                        html.H3("📂 Carga de datos"),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Arrastra o selecciona un archivo CSV/Excel"]),
                            className="upload-box"
                        ),
                        html.Div("O usa el archivo de ejemplo: ejemplo.csv", className="nota"),

                        html.Hr(),
                        html.H3("⚙️ Configuración del gráfico"),
                        dcc.Dropdown(
                            id="chart-type",
                            options=[
                                {"label": "X̅ - R", "value": "X_R"},
                                {"label": "X̅ - S", "value": "X_S"}
                            ],
                            value="X_R",
                            className="dropdown"
                        ),

                        html.Div([
                            html.Label("Nivel de significancia (α):"),
                            dcc.Input(
                                id='alpha-input',
                                type='number',
                                placeholder='0.05',
                                min=0.001, max=0.2, step=0.001,
                                value=None,  # si el usuario no pone nada, usará el default
                                style={'width': '100px'}
                            ),
                        ], style={'marginTop': '10px', 'display': 'inline-block', 'marginLeft': '10px'}),


                        html.Br(),
                        html.H4("Límites de especificación (opcionales)"),
                        html.Label("LSL"),
                        dcc.Input(id="lsl", type="number", placeholder="Ej: 9.5", className="input"),
                        html.Label("USL"),
                        dcc.Input(id="usl", type="number", placeholder="Ej: 10.5", className="input"),

                        html.Button("Usar límites ingresados", id="use-manual-limits", n_clicks=0, className="btn-accion"),
                        html.Hr(),
                        
                        html.Label("Indicador de capacidad a evaluar:"),
                        dcc.Dropdown(
                            id='capability_metric',
                            options=[
                                {'label': 'Corto plazo (Cp/Cpk)', 'value': 'short'},
                                {'label': 'Largo plazo (Pp/Ppk)', 'value': 'long'}
                            ],
                            value='short',  # valor por defecto
                            clearable=False,
                            style={'width': '90%', 'margin-bottom': '10px'}
                        ),

                        html.H4("Opciones de visualización"),
                        dcc.Checklist(
                            id="show-points",
                            options=[{"label": "Mostrar valores de subgrupo", "value": "show"}],
                            value=["show"],
                            className="checklist"
                        ),
                        html.Div(id="n-info", className="info"),

                        html.Hr(),
                        html.H3("📊 Análisis estadístico adicional", className="titulo-seccion"),

                        html.Div([
                            html.H4("Prueba de Normalidad (Shapiro–Wilk)"),
                            html.Button("🔍 Realizar prueba de normalidad", 
                                id="btn-normalidad", 
                                n_clicks=0, 
                                className="btn-accion btn-normalidad"),
                        ], className="bloque-analisis"),
                    ]
                ),

                # Columna derecha
                html.Div(
                    className="contenido",
                    children=[
                        html.H3("📊 Previsualización de datos"),
                        dash_table.DataTable(
                            id="table-data",
                            page_size=8,
                            style_table={"overflowX": "auto"}
                        ),
                        html.Hr(),
                        html.H3("📈 Gráfico de control"),
                        dcc.Loading(
                            dcc.Graph(
                                id="control-chart",
                                figure={
                                    "layout": {
                                        "paper_bgcolor": "rgb(30, 30, 30)",
                                        "plot_bgcolor": "rgb(30, 30, 30)",
                                        "font": {"color": "white"}
                                    }
                                },
                                style={"backgroundColor": "rgb(30, 30, 30)"}
                            ),
                            type="default"
                        ),

                        # Tarjetas con indicadores del proceso
                        html.Div(id="indicators", className="indicadores"),
                        html.Div(id="alerts", className="alertas"),

                        #        SECCIÓN DE RESULTADOS DE NORMALIDAD 
                        html.Div(className="seccion-normalidad", children=[
                            html.Hr(),
                            html.H3("📈 Resultados de la Prueba de Normalidad (Shapiro–Wilk)", 
                                id='titulo-normalidad',
                                className="titulo-seccion",
                                style={'display': 'none'}),

                            dcc.Graph(id='grafico-normalidad', style={'marginTop': 20, 'display': 'none'}),

                            html.Div(id='resultado-normalidad', 
                                style={'marginTop': 10, 'fontSize': 18, 'textAlign': 'center', 'display': 'none'})
                        ]),
                    ]
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("main-container", "className"),
    Output("toggle-theme", "children"),
    Input("toggle-theme", "n_clicks"),
    State("main-container", "className"),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_class):
    if current_class == "modo-claro":
        return "modo-oscuro", "☀️"
    else:
        return "modo-claro", "🌙"


# ------------------------------ Helpers ------------------------------

def parse_contents(contents, filename):
    import csv
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        text = decoded.decode('utf-8', errors='ignore')
        # tomar varias líneas para una mejor detección
        sample = '\n'.join(text.splitlines()[:10])

        if 'csv' in filename.lower():
            # Intentar detectar delimitador usando muestra más amplia
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except Exception:
                delimiter = ','
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter)
        elif 'xls' in filename.lower() or 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            # Intentar leer como CSV de respaldo
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
    except Exception as e:
        raise e

    return df


def load_example():
    try:
        df = pd.read_csv('data/ejemplo.csv')
        return df
    except Exception:
        df = pd.DataFrame({
            'Subgrupo':[1,2,3],
            'X1':[10.2,10.5,9.7],
            'X2':[9.8,10.6,9.8],
            'X3':[10.1,10.4,9.6],
            'X4':[10.3,10.7,9.9],
            'X5':[9.9,10.3,10.0]
        })
        return df

def compute_control_limits(df, chart_type='X_R', alpha=None):
    # alpha por defecto
    if alpha is None or alpha <= 0:
        alpha = 0.05
    Z = norm.ppf(1 - alpha/2)

    # columnas numéricas de medición (ignorar 'Subgrupo' si existe)
    meas = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Subgrupo' in meas:
        meas.remove('Subgrupo')
    if len(meas) == 0:
        raise ValueError("No se encontraron columnas numéricas con mediciones")

    n = len(meas)      # tamaño del subgrupo (mediciones por fila)
    k = df.shape[0]    # número de subgrupos

    # Estadísticas por subgrupo
    subgroup_means = df[meas].mean(axis=1).values         # array tamaño k
    subgroup_ranges = (df[meas].max(axis=1) - df[meas].min(axis=1)).values
    subgroup_std = df[meas].std(axis=1, ddof=1).values    # sd dentro de cada subgrupo

    # Estadísticas globales
    Xbar_bar = np.mean(subgroup_means) if k > 0 else float('nan')
    Rbar = np.mean(subgroup_ranges) if k > 0 else float('nan')
    Sbar = np.mean(subgroup_std) if k > 0 else float('nan')

    limits = {}

    # variabilidad entre promedios (sd de los promedios)
    s_promedios = np.std(subgroup_means, ddof=1) if k > 1 else 0.0
    UCLx = Xbar_bar + Z * s_promedios
    LCLx = Xbar_bar - Z * s_promedios
    limits['Xbar'] = {'CL': Xbar_bar, 'UCL': UCLx, 'LCL': LCLx}

    # estimación sigma_within (variabilidad dentro de subgrupos) - para Cp/Cpk corto plazo
    # uso la raíz cuadrada ponderada por grados de libertad:
    if k * (n - 1) > 0:
        dof_total = k * (n - 1)
        sigma_within = math.sqrt(np.sum((n - 1) * (subgroup_std ** 2)) / dof_total)
    else:
        sigma_within = float('nan')

    # límites para R o S usando la variabilidad entre subgrupos (Z * sd de rangos/desv.)
    if chart_type == 'X_R':
        R_std = np.std(subgroup_ranges, ddof=1) if k > 1 else 0.0
        limits['R'] = {'CL': Rbar, 'UCL': Rbar + Z * R_std, 'LCL': max(Rbar - Z * R_std, 0)}
        sigma_est = sigma_within   # usar sigma_within para Cp/Cpk (más coherente)
    else:  # X_S
        S_std = np.std(subgroup_std, ddof=1) if k > 1 else 0.0
        limits['S'] = {'CL': Sbar, 'UCL': Sbar + Z * S_std, 'LCL': max(Sbar - Z * S_std, 0)}
        sigma_est = sigma_within if not math.isnan(sigma_within) else Sbar

    return {
        'n': n,
        'k': k,
        'meas_cols': meas,
        'subgroup_means': subgroup_means,
        'subgroup_ranges': subgroup_ranges,
        'subgroup_std': subgroup_std,
        'limits': limits,
        'Xbar_bar': Xbar_bar,
        'Rbar': Rbar,
        'Sbar': Sbar,
        'sigma_est': sigma_est,
        'alpha': alpha,
        'Z': Z
    }

# ------------------------------ Callbacks ------------------------------

@app.callback(
    Output('table-data', 'data'),
    Output('table-data', 'columns'),
    Output('n-info', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        msg = f"Datos cargados: {filename} - {df.shape[0]} filas x {df.shape[1]} columnas."
    else:
        df = load_example()
        msg = 'Usando archivo de ejemplo (data/ejemplo.csv)'
    columns=[{'name':c,'id':c} for c in df.columns]
    return df.to_dict('records'), columns, msg


@app.callback(
    Output('control-chart', 'figure'),
    Output('indicators','children'),
    Output('alerts','children'),
    Input('upload-data', 'contents'),
    Input('chart-type','value'),
    Input('use-manual-limits','n_clicks'),
    State('upload-data','filename'),
    State('lsl','value'),
    State('usl','value'),
    Input('show-points','value'),
    Input('alpha-input', 'value'),
    Input('main-container', 'className'),
    Input('capability_metric', 'value')
)
def update_chart(contents, chart_type, n_clicks, filename, lsl, usl, show_points, alpha, theme_class, capability_metric):
    # load data
    if contents is not None:
        df = parse_contents(contents, filename)
    else:
        df = load_example()

    try:
        comp = compute_control_limits(df, chart_type, alpha)
    except Exception as e:
        fig = go.Figure()
        return fig, '', f'Error al calcular límites: {str(e)}'

    n = comp['n']
    means = comp['subgroup_means']
    x = list(range(1, len(means)+1))
    limits = comp['limits']

    fig = go.Figure()

    show_list = show_points or []

    # plot Xbar
    fig.add_trace(go.Scatter(x=x, y=means, mode='lines+markers' if 'show' in show_list else 'lines', name='X̅ por subgrupo'))
    fig.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['Xbar']['CL'], limits['Xbar']['CL']], mode='lines', name='CL X̅', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['Xbar']['UCL'], limits['Xbar']['UCL']], mode='lines', name='UCL X̅', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['Xbar']['LCL'], limits['Xbar']['LCL']], mode='lines', name='LCL X̅', line=dict(color='gold')))

    if usl is not None:
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[usl, usl], mode='lines', name='USL', line=dict(color='#6BBE45', dash='solid'), hovertemplate='USL: %{y:.2f}<extra></extra>'))

    if lsl is not None:
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[lsl, lsl], mode='lines', name='LSL', line=dict(color='#6BBE45', dash='solid'), hovertemplate='LSL: %{y:.2f}<extra></extra>'))

    ooc = [i+1 for i, val in enumerate(means) if (val > limits['Xbar']['UCL'] or val < limits['Xbar']['LCL'])]
    if len(ooc) > 0:
        fig.add_trace(go.Scatter(
            x=ooc,
            y=[means[i-1] for i in ooc],
            mode='markers',
            name='X̅ fuera de control',
            marker=dict(color='red', size=10, symbol='x')
        ))

    # Construir resumen estilo Minitab
    try:
        CLx = limits['Xbar']['CL']
        UCLx = limits['Xbar']['UCL']
        LCLx = limits['Xbar']['LCL']

        ct = str(chart_type).strip().upper()
        if ct == 'X_R':
            label2 = 'R'
            CL2 = limits.get('R', {}).get('CL', None)
            UCL2 = limits.get('R', {}).get('UCL', None)
            LCL2 = limits.get('R', {}).get('LCL', None)
            extra_rows = [
                html.Tr([html.Td("Subgroup size (n)"), html.Td(f"{n}")]),
                html.Tr([html.Td("X̅̅"), html.Td(f"{comp['Xbar_bar']:.4f}")]),
                html.Tr([html.Td("R̄"), html.Td(f"{comp['Rbar']:.4f}")])
            ]
            sigma = comp.get('sigma_est', None)
        else:
            label2 = 'S'
            CL2 = limits.get('S', {}).get('CL', None)
            UCL2 = limits.get('S', {}).get('UCL', None)
            LCL2 = limits.get('S', {}).get('LCL', None)
            extra_rows = [
                html.Tr([html.Td("Subgroup size (n)"), html.Td(f"{n}")]),
                html.Tr([html.Td("X̅̅"), html.Td(f"{comp['Xbar_bar']:.4f}")]),
                html.Tr([html.Td("S̄"), html.Td(f"{comp['Sbar']:.4f}")])
            ]
            sigma = comp.get('sigma_est', None)

        Relacion = (UCLx - CLx) / (CLx - LCLx) if (CLx - LCLx) != 0 else float('nan')

        # Cálculo de indicadores de capacidad SIN constantes

        Cp = Cpk = Pp = Ppk = None
        sigma_within = sigma_total = None

        if lsl is not None and usl is not None:
            try:
                all_vals = df[comp['meas_cols']].values.flatten()
                mean_all = np.nanmean(all_vals)

                # σ dentro de los subgrupos
                subgroup_std = comp['subgroup_std']
                n = comp['n']
                k = comp['k']
                dof_total = k * (n - 1)
                sigma_within = np.sqrt(np.sum((n - 1) * subgroup_std**2) / dof_total)

                # σ total de todos los datos
                sigma_total = np.std(all_vals, ddof=1)

                # CAPACIDAD (corto plazo)
                if sigma_within > 0:
                    Cp = (usl - lsl) / (6 * sigma_within)
                    Cpk = min((usl - mean_all) / (3 * sigma_within), (mean_all - lsl) / (3 * sigma_within))

                # DESEMPEÑO (largo plazo)
                if sigma_total > 0:
                    Pp = (usl - lsl) / (6 * sigma_total)
                    Ppk = min((usl - mean_all) / (3 * sigma_total), (mean_all - lsl) / (3 * sigma_total))
            except Exception as e:
                print("Error al calcular Cp/Cpk/Pp/Ppk:", e)

        table_rows = [html.Tr([html.Th("Indicador"), html.Th("Valor")])]
        table_rows += extra_rows
        table_rows += [
            html.Tr([html.Td("X̅ Media (CL)"), html.Td(f"{CLx:.3f}")]),
            html.Tr([html.Td("UCL X̅"), html.Td(f"{UCLx:.3f}")]),
            html.Tr([html.Td("LCL X̅"), html.Td(f"{LCLx:.3f}")]),
            html.Tr([html.Td("Relacion UCL/LCL"), html.Td(f"{Relacion:.3f}")]),
            html.Tr([html.Td(f"{label2} Media (CL)"), html.Td(f"{CL2:.3f}" if CL2 is not None else '-')]),
            html.Tr([html.Td(f"UCL {label2}"), html.Td(f"{UCL2:.3f}" if UCL2 is not None else '-')]),
            html.Tr([html.Td(f"LCL {label2}"), html.Td(f"{LCL2:.3f}" if LCL2 is not None else '-')]),
            html.Tr([html.Td("Nivel de significancia (α)"), html.Td(f"{comp['alpha']:.3f}")]),
            html.Tr([html.Td("Valor crítico Zα/2"), html.Td(f"{comp['Z']:.3f}")])
        ]

        if any(v is not None for v in [Cp, Cpk, Pp, Ppk]):
            table_rows += [
                html.Tr([html.Th("", colSpan=2, style={'backgroundColor':'#transparent'})]),
                html.Tr([html.Th("📈 Indicadores de Capacidad", colSpan=2, style={'textAlign':'center'})]),
                html.Tr([html.Td("σ (dentro de subgrupos)"), html.Td(f"{sigma_within:.5f}" if sigma_within else "—")]),
                html.Tr([html.Td("σ (total del proceso)"), html.Td(f"{sigma_total:.5f}" if sigma_total else "—")]),
                html.Tr([html.Td("Cp"), html.Td(f"{Cp:.4f}" if Cp else "—")]),
                html.Tr([html.Td("Cpk"), html.Td(f"{Cpk:.4f}" if Cpk else "—")]),
                html.Tr([html.Td("Pp"), html.Td(f"{Pp:.4f}" if Pp else "—")]),
                html.Tr([html.Td("Ppk"), html.Td(f"{Ppk:.4f}" if Ppk else "—")]),
           ]

        # === Evaluación interpretativa de capacidad (según selección del usuario) ===
        capacidad_alert = ""
        color_capacidad = "#000000"
        clase_interpretacion = "interpretacion"  # clase base

        # Determinar indicador base según selección
        if capability_metric == 'short':  # corto plazo
            indicador = Cpk
            nombre = "Cpk"
        else:  # largo plazo
            indicador = Ppk
            nombre = "Ppk"

        if indicador is not None:
            if indicador >= 1.33:
                capacidad_alert = f"✅ El proceso es capaz ({nombre} = {indicador:.2f})."
                color_capacidad = "#2E8B57"  # verde
                clase_interpretacion += " interpretacion-green"
            elif indicador >= 1.00:
                capacidad_alert = f"⚠️ El proceso es marginal ({nombre} = {indicador:.2f})."
                color_capacidad = "#FF8C00"  # naranja
                clase_interpretacion += " interpretacion-orange"
            else:
                capacidad_alert = f"❌ El proceso NO es capaz ({nombre} = {indicador:.2f})."
                color_capacidad = "#DC143C"  # rojo
                clase_interpretacion += " interpretacion-red"
        else:
            capacidad_alert = "ℹ️ No hay datos suficientes para evaluar la capacidad."
            color_capacidad = "#444"
            clase_interpretacion += ""  # sin color

        interpretacion_div = html.Div(
            capacidad_alert,
            className=clase_interpretacion,
            style={'color': color_capacidad}
        )

        indicators_table = html.Div([
            html.H4("📊 Resumen del gráfico", className='titulo-resumen'),
            interpretacion_div,
            html.Table(table_rows, className='tabla-resumen')
        ])

    except Exception as e:
        indicators = html.Div(f"⚠️ Error al generar resumen: {e}", style={'color':'red'})

    # mark out-of-control points for Xbar
    ooc = [i+1 for i,val in enumerate(means) if (val > limits['Xbar']['UCL'] or val < limits['Xbar']['LCL'])]

    # prepare second chart defaults
    ooc_r = []
    ooc_s = []

    # secondary chart: R or S
    indicators_secondary = html.Div()
    if chart_type == 'X_R':
        r = comp['subgroup_ranges']
        y2 = r
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers' if 'show' in show_list else 'lines', name='R por subgrupo'))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['R']['CL'], limits['R']['CL']], mode='lines', name='CL R', line=dict(dash='dash')))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['R']['UCL'], limits['R']['UCL']], mode='lines', name='UCL R', line=dict(color='gold')))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['R']['LCL'], limits['R']['LCL']], mode='lines', name='LCL R', line=dict(color='gold')))
        ooc_r = [i+1 for i,val in enumerate(y2) if (val > limits['R']['UCL'] or val < limits['R']['LCL'])]
        if len(ooc_r) > 0:
            fig2.add_trace(go.Scatter(x=ooc_r, y=[y2[i-1] for i in ooc_r], mode='markers', name='R fuera de control', marker=dict(color='red', size=10, symbol='x')))

        fig.update_layout(title='Gráfico X̅ (superior) y R (inferior)')
        indicators_secondary = html.Div([
            dcc.Graph(figure=fig2, style={'height':'300px'})
        ])


    else:
        s = comp['subgroup_std']
        y2 = s
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers' if 'show' in show_list else 'lines', name='S por subgrupo'))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['S']['CL'], limits['S']['CL']], mode='lines', name='CL S', line=dict(dash='dash')))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['S']['UCL'], limits['S']['UCL']], mode='lines', name='UCL S', line=dict(color='gold')))
        fig2.add_trace(go.Scatter(x=[min(x),max(x)], y=[limits['S']['LCL'], limits['S']['LCL']], mode='lines', name='LCL S', line=dict(color='gold')))
        ooc_s = [i+1 for i,val in enumerate(y2) if (val > limits['S']['UCL'] or val < limits['S']['LCL'])]
        if len(ooc_s) > 0:
            fig2.add_trace(go.Scatter(x=ooc_s, y=[y2[i-1] for i in ooc_s], mode='markers', name='S fuera de control', marker=dict(color='red', size=10, symbol='x')))

        fig.update_layout(title='Gráfico X̅ (superior) y S (inferior)')
        indicators_secondary = html.Div([
            dcc.Graph(figure=fig2, style={'height':'300px'})
        ])

    # Alerts text
    alerts = []
    if len(ooc) > 0:
        alerts.append(html.Div(f'Puntos fuera de control en X̅: {ooc}'))

    if chart_type == 'X_R' and len(ooc_r) > 0:
        alerts.append(html.Div(f'Puntos fuera de control en R: {ooc_r}'))
    if chart_type == 'X_S' and len(ooc_s) > 0:
        alerts.append(html.Div(f'Puntos fuera de control en S: {ooc_s}'))

    alert_comp = html.Div(alerts) if len(alerts)>0 else html.Div('Sin puntos fuera de control detectados.', style={'color':'green'})

    # === Mostrar puntos individuales si está activo ===
    show_points_active = "show" in show_points if show_points else False

    if show_points_active and 'Valor' in df.columns:
        try:
            fig.add_trace(go.Scatter(
                x=df["Subgrupo"],
                y=df["Valor"],
                mode='markers',
                marker=dict(color='blue', size=6, opacity=0.7),
                name='Valores individuales'
            ))
        except Exception as e:
            print("Error al mostrar puntos individuales:", e)

    # format main figure and apply theme
    fig.update_layout(height=450, xaxis_title='Subgrupo', yaxis_title='Valor')

    if theme_class == "modo-oscuro":
        theme_layout = dict(
            plot_bgcolor="#1b1b1b",
            paper_bgcolor="#1b1b1b",
            font_color="#e0e0e0",
            title_font_color="#ffffff",
            legend_bgcolor="#252525",
            legend_font_color="#e0e0e0",
            xaxis=dict(gridcolor="#444", color="#e0e0e0"),
            yaxis=dict(gridcolor="#444", color="#e0e0e0")
        )
    else:
        theme_layout = dict(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#000000",
            title_font_color="#000000",
            legend_bgcolor="#f5f5f5",
            legend_font_color="#000000",
            xaxis=dict(gridcolor="#ddd", color="#000000"),
            yaxis=dict(gridcolor="#ddd", color="#000000")
        )

    fig.update_layout(**theme_layout)
    try:
        fig2.update_layout(**theme_layout)
    except:
        pass

    indicators_final = html.Div([
        indicators_secondary,  
        html.Hr(),
        indicators_table,      
        html.Hr()
    ])

    return fig, indicators_final, alert_comp

@app.callback(
    [Output('titulo-normalidad', 'style'),
     Output('resultado-normalidad', 'children'),
     Output('resultado-normalidad', 'style'),
     Output('grafico-normalidad', 'figure'),
     Output('grafico-normalidad', 'style')],
    Input('btn-normalidad', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('main-container', 'className'),
    prevent_initial_call=True
)
def verificar_normalidad(n_clicks, contents, filename, theme_class):
    oculto = {'display': 'none'}
    
    if not n_clicks or not contents:
        return oculto, dash.no_update, oculto, dash.no_update, oculto

    try:
        # Leer y procesar datos
        df = parse_contents(contents, filename)
        resultados = compute_control_limits(df)
        subgroup_means = resultados['subgroup_means']

        mu = np.mean(subgroup_means)
        sigma = np.std(subgroup_means, ddof=1)

        stat, p_value = shapiro(subgroup_means)
        if p_value > 0.05:
            mensaje = f"✅ Estadístico W = {stat:.4f}. Los promedios de los subgrupos parecen normales (p = {p_value:.4f})"
        else:
            mensaje = f"⚠️ Estadístico W = {stat:.4f}. Los promedios de los subgrupos no parecen normales (p = {p_value:.4f})"

        # Crear gráfico
        x = np.linspace(min(subgroup_means), max(subgroup_means), 100)
        pdf = norm.pdf(x, mu, sigma)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=subgroup_means,
            histnorm='probability density',
            name='Promedios de subgrupos',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pdf,
            mode='lines',
            name='Curva normal teórica',
            line=dict(color='red', width=2)
        ))

        # Tema adaptable
        if theme_class == "modo-oscuro":
            template = 'plotly_dark'
            font_color = 'white'
        else:
            template = 'plotly_white'
            font_color = 'black'

        fig.update_layout(
            title='Distribución de promedios de subgrupos vs Curva Normal',
            xaxis_title='Promedio del subgrupo',
            yaxis_title='Densidad',
            template=template,
            font=dict(color=font_color)
        )

        msg_style = {'color': font_color, 'display': 'block', 'marginTop': '10px'}
        visible = {'display': 'block'}
        return visible, mensaje, msg_style, fig, visible

    except Exception as e:
        # Si ocurre un error, ocultar el gráfico y mostrar mensaje
        oculto = {'display': 'none'}
        return oculto, f"❌ Error al procesar los datos: {e}", oculto, dash.no_update, oculto

@app.callback(
    [
      Output('grafico-normalidad', 'figure', allow_duplicate=True),
      Output('resultado-normalidad', 'style', allow_duplicate=True)
    ],
    Input('main-container', 'className'),
    State('grafico-normalidad', 'figure'),
    State('resultado-normalidad', 'children'),
    prevent_initial_call=True
)
def actualizar_tema_normalidad(theme_class, current_fig, current_text):
    """
    Ajusta el tema del gráfico de normalidad y el color del texto del resultado
    cuando el usuario cambia entre modo claro/oscuro.
    """
    # Si no hay gráfico ni texto, no hacemos nada
    if not current_fig and not current_text:
        raise dash.exceptions.PreventUpdate

    # Si hay figura la ajustamos
    fig = None
    if current_fig:
        fig = go.Figure(current_fig)
        if theme_class and "oscuro" in theme_class.lower():
            fig.update_layout(
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e'
            )
        else:
            fig.update_layout(
                template='plotly_white',
                font=dict(color='black'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )

    # Ajustamos estilo del texto si ya existe
    if current_text:
        if theme_class and "oscuro" in theme_class.lower():
            text_style = {'color': 'white', 'display': 'block', 'marginTop': '10px', 'textAlign': 'center'}
        else:
            text_style = {'color': 'black', 'display': 'block', 'marginTop': '10px', 'textAlign': 'center'}
    else:
        text_style = {'display': 'none'}

    # devolver figura (o dash.no_update si no existía) y el estilo del texto
    return (fig if fig is not None else dash.no_update), text_style


if __name__ == '__main__':

    app.run_server(host="0.0.0.0", port=8050)



