import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import plotly.express as px
import plotly.graph_objects as go
import random
import geopandas as gpd
from shapely.geometry import Point
import scipy
from scipy.spatial.distance import cdist
import pulp
from pulp import LpVariable, LpMinimize, LpProblem, lpSum,LpBinary
import time
import math


###
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
brazil_polygon = world[world.name == 'Brazil']['geometry'].values[0]

def generate_grid_points(max_lat, min_lat, max_lon, min_lon, distance = 0.090090):
    points = []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            point_geom = Point(lon, lat)

            # Verificar se o ponto está dentro do polígono do Brasil
            if point_geom.within(brazil_polygon):
                points.append((lat, lon))

            lon += distance
        lat += distance
    return points

def veto_points(points, base_data, distancia_em_km): #altera aqui pra mudar distancia de veto

  """
  Essa função retorna um grid de pontos espaçados de uma distância a determinada, a partir de coordenadas máximas e mínimas do grid.

  :max_lat:float
  :min_lat:float
  :max_long:float
  :min_long:float
  :distancia_em_km:float

  """
  distancia_em_graus = (distancia_em_km*0.090090)/ 10
  vetoed_points = []
  for point in points:
    lat, lon = point
    point_geom = Point(lon, lat)

    # Verificar se o ponto está dentro do polígono do Brasil
    if not point_geom.within(brazil_polygon):
        continue

    # Verificar a distância para os pontos da base
    distances = np.sqrt((base_data[:, 0] - lat)**2 + (base_data[:, 1] - lon)**2)
    min_distance = min(distances)

    if min_distance <= distancia_em_graus:
        vetoed_points.append(point)

  return vetoed_points


###

st.set_page_config(
    page_title="GTM by Pragmatis",
        layout="wide",
    initial_sidebar_state="expanded"

)

css = """
 footer {
    visibility : hidden;
 }

.stButton > button:hover {
    cursor: pointer;
}

"""

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


st.title('GTM - Setorização de Clientes')

tab1, tab2, tab3 = st.tabs(["Página Principal", "Visualização", "Sobre"])

with tab1:

    st.write("Setup:")
        
    form = st.form(key="Configurações")

    with form:
        col1, col2 = st.columns([1,1])

        with col1:

            st.write("Logística:")
            raio_max = st.number_input(
                "Raio Máximo de Atendimento (Km):",
                value=700,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i")
            
            modelo_pontos_candidatos = ["Quadrantes com Veto", "Pontos de Clientes"]

            
            bg_shape = st.radio(
            "Pontos Candidatos:",
            options=modelo_pontos_candidatos,
            key="bg_shape")
            
            if bg_shape == "Quadrantes com Veto":
                modelo_grid = 1

            distancia_pontos_grid = st.number_input(
                "Distancia entre Pontos (Km):",
                value=10,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i" )

            distancia_de_veto = st.number_input(
                "Distancia de Veto (Km):",
                value=10,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i" )
            
            bg_shape_options_2 = [ " == P"," <= P"]
            rd_restr = st.radio(
            "Número de Vendedores:",
            options=bg_shape_options_2)

            if rd_restr == " <= P":
                p_menor_igual = 1
                
            else:
                p_menor_igual = 0

            configurado = st.form_submit_button("Configurar")

        with col2:

            st.write("Restrições:")

            breakeven = st.number_input(
                "Breakeven (R$)",
                value=5500000,
                min_value=1,
                max_value=9999999,
                step=1)

            tempo_limite= st.number_input(
                "Tempo limite de otimização:",
                value=300,
                min_value=1,
                max_value=9999999,
                step=1)

            com_tempo_limite_de_processamente = 1

            st.write("Full time equivalent (FTE):")

            FTE_min = st.number_input(
                "FTE Mínimo:",
                value=0.60,
                min_value=0.0,
                max_value=2.0,
                step=0.05)

            FTE_backoffice = st.number_input(
                "FTE Backoffice:",
                value= 0.41,
                min_value=0.0,
                max_value=2.0,
                step=0.05)
            
            FTE_max = st.multiselect('FTE Máximo:',[0.95, 1, 1.05,1.10,1.15,1.20,1.25], [0.95, 1, 1.05,1.10,1.25])
            
            

    base_de_clientes = st.file_uploader('Insira a Base de Clientes')

    if base_de_clientes:

        df_base_clientes = pd.read_excel(base_de_clientes)
        df_p = df_base_clientes[['SubCoord_TOBE', 'p']].drop_duplicates()
        df_p['p'] = df_p['p'].astype(int)

        lista_subcoords = list(set(list(df_base_clientes['SubCoord_TOBE'])))
        subcoords_ = st.multiselect('Subcoords',sorted(lista_subcoords),lista_subcoords[0:4])
        simular = st.button("Simular")

        if simular:

            geral = pd.DataFrame({'lat':[],'long':[],'latitude_centro':[], 'longitude_centro':[], 'cluster_centro':[], 'SubCoord':[]})
            cont = 1
            sub = []
            status = []
            aux = 0

            for subcoords in subcoords_:

                st.write(cont, "/",len(subcoords_), ' | ', subcoords)

                cont=cont +1

                DF_CLIENTS_SUBCOORD = df_base_clientes.query(f'SubCoord_TOBE =="{subcoords}"')
                DF_P = df_p.query(f'SubCoord_TOBE =="{subcoords}"')
                data = DF_CLIENTS_SUBCOORD[['lat', 'long']].values

                # Definir os limites de lat/long permitidos na região da base
                max_lat_base = max(data[:, 0])
                min_lat_base = min(data[:, 0])
                max_lon_base = max(data[:, 1])
                min_lon_base = min(data[:, 1])

                # Gerar grid:
                grid_points = generate_grid_points(max_lat_base, min_lat_base, max_lon_base, min_lon_base)
                # Criar um DataFrame pandas com os pontos do grid:
                grid_df = pd.DataFrame(grid_points, columns=['lat', 'long'])
                # Aplicar o veto para eliminar pontos distantes da base:
                vetoed_points = veto_points(grid_points, data,distancia_de_veto)
                # Criar um DataFrame pandas com os pontos do grid após o veto:
                vetoed_grid_df = pd.DataFrame(vetoed_points, columns=['lat', 'long'])

                # MODELANDO PARAMETROS DO MODELO:
                # Dados de entrada (somente latitude e longitude)

                points_to_be_served_2 = DF_CLIENTS_SUBCOORD[['Nome rede_ Ajustado','Nome PDV_Ajustado','lat', 'long','FTE_Atend_Visita','Visitas presenciais','ROL']].reset_index(drop=True)
                points_to_be_served = DF_CLIENTS_SUBCOORD[['Nome rede_ Ajustado','Nome PDV_Ajustado','lat', 'long','FTE_Atend_Visita','Visitas presenciais','ROL']].reset_index(drop=True)[['lat', 'long']]
                FTE_Atendimento = list(points_to_be_served_2['FTE_Atend_Visita'])
                ROL = list(points_to_be_served_2['ROL'])
                Visita = list(points_to_be_served_2.fillna(0)['Visitas presenciais'])        

                network_dict = {}
                for _, row in points_to_be_served_2.iterrows():
                    network_name = row['Nome rede_ Ajustado']
                    pdv_name = row['Nome PDV_Ajustado']
                    if network_name not in network_dict:
                        network_dict[network_name] = []
                    network_dict[network_name].append(pdv_name)                

                #ESCOLHA DE POSSÍVEIS LOCAIS CANDIDATOS:
                if modelo_grid == 1:
                    potential_locations = vetoed_grid_df[['lat', 'long']]
                else:
                    potential_locations = points_to_be_served_2[['lat','long']]

                # Cálculo das distâncias entre pontos e potenciais localizações
                distance_matrix = cdist(points_to_be_served, potential_locations, metric='euclidean') * 111.11*1.23

                st.write("--> CDD = #",len(potential_locations))
                st.write("--> PDV = #",len(points_to_be_served))

                c = distance_matrix

                p = list(DF_P['p'])[0]

                st.write("--> FTE_Atendimento =",round(sum(FTE_Atendimento),2))

                st.write("--> p  =",p)

                num_tentativas = 0

                status_ =3

                while (status_ != 1 and num_tentativas <= (len(sorted(FTE_max)) -1)):

                #  print(f'Testando com FTE = {fte_suporte +0.41}...')

                    vec_suporte = sorted(FTE_max)

                    fte_suporte = vec_suporte[num_tentativas] - FTE_backoffice
                    # Defining the problem:
                    prob = LpProblem("p-center", LpMinimize)

                    m = len(points_to_be_served)
                    n = len(potential_locations)

                    x = LpVariable.dicts("x", [(i, j) for i in range(m) for j in range(n)], 0, 1, LpBinary)
                    y = LpVariable.dicts("y", [j for j in range(n)], 0, 1, LpBinary)

                    # Define objective function
                    prob += lpSum([c[i][j] * x[(i,j)] for i in range(m) for j in range(n)])

                    # Constraints:
                    for i in range(m):
                        prob += lpSum([x[(i, j)] for j in range(n)]) == 1

                    for i in range(m):
                        for j in range(n):
                            prob += x[(i, j)] <= y[j] # Ensure distance is less than or equal to Z
                            prob += c[i][j] * x[(i,j)] <= x[(i,j)]*raio_max

                    if p_menor_igual == 1:
                        prob += lpSum([y[j] for j in range(n)]) <= p
                    else:
                        prob += lpSum([y[j] for j in range(n)]) == p


                    for j in range(n):

                        prob += lpSum(x[(i, j)]* Visita[i]*( 0.25  + (c[i][j] / 65)) for i in range(m) )  / 220  + lpSum(x[(i, j)] *FTE_Atendimento[i] for i in range(m)) <= fte_suporte*y[j]
                        prob += lpSum(x[(i, j)]* Visita[i]*( 0.25  + (c[i][j] / 65)) for i in range(m) ) / 220  + lpSum(x[(i, j)] *FTE_Atendimento[i] for i in range(m)) >= (FTE_min - FTE_backoffice)*y[j]

                        prob += lpSum(x[(i, j)]* ROL[i] for i in range(m) )  >= breakeven*y[j]

                    for network, members in network_dict.items():
                        network_members = [i for i, row in points_to_be_served_2.reset_index(drop=True).iterrows() if row['Nome rede_ Ajustado'] == network]
                        if network_members:
                            for nm in network_members:
                                for nms in network_members:
                                    for j in range(n):
                                        prob += x[(nms, j)] == x[(nm, j)]

                    tempo_inicial = time.time()

                    if com_tempo_limite_de_processamente == 1:
                        with st.spinner('Aguarde a otimização...'):
                            status_ = prob.solve(pulp.PULP_CBC_CMD(timeLimit =tempo_limite, msg=1))

                        tempo_final = time.time()

                    else:
                        status_ = prob.solve()

                        tempo_final = time.time()

                        num_tentativas = num_tentativas+1

                if status_ == 1:
                    st.write(f'--> TEM SOLUÇÃO COM FTE = {fte_suporte+0.41} ')

                    st.write("--> tempo de solução =",round(tempo_final-tempo_inicial,0))

                    st.write("\n")

                    for j in range(n):
                        if y[j].value() ==1:

                            st.write(f'CLUSTER - {j}')

                            st.write(f'--> FTE Total: ',round(sum(x[(i, j)].value() * Visita[i] * (0.25 + (c[i][j] / 65)) for i in range(m)) / 220 + sum(x[(i, j)].value() * FTE_Atendimento[i] for i in range(m)) + 0.41 ,2))

                            st.write(f'--> # de PDV:',  round(sum(x[(i, j)].value() * 1 for i in range(m) ) ,0 ) )

                            Lat_ = sum(x[(i, j)].value()*ROL[i]*list(points_to_be_served_2['lat'])[i] for i in range(m)) /sum(ROL[i]*x[(i, j)].value() for i in range(m))
                            Lon_ = sum(x[(i, j)].value()*ROL[i]*list(points_to_be_served_2['long'])[i] for i in range(m)) /sum(ROL[i]*x[(i, j)].value() for i in range(m))

                            fte_ = round(sum(x[(i, j)].value() * Visita[i] * (0.25 + ( ( math.sqrt( (Lat_ -list(points_to_be_served_2['lat'])[i])**2 + ( Lon_ -list(points_to_be_served_2['long'])[i])**2)) / 65)) for i in range(m)) / 220 + sum(x[(i, j)].value()*FTE_Atendimento[i] for i in range(m)) + 0.41 ,2)

                            st.write(f'--> FTE CM: {fte_} (Lat = {round(Lat_,4)},Lon = {round(Lon_,4)})')

                            st.write(f'--> ROL Total: ', round(sum(x[(i, j)].value() * ROL[i] for i in range(m) ) ,0 ) )

                            lista_vendedor = []
                            lista_cliente = []

                    for j in range(n):
                        for i in range(m):
                            if x[(i, j)].value():
                                lista_cliente.append(i)
                                lista_vendedor.append(j)

                    vendedor_df = pd.DataFrame({'Vendedor':lista_vendedor,'Cliente':lista_cliente})
                    vendedor_df = vendedor_df = vendedor_df.sort_values('Cliente').reset_index(drop = True)

                    points_to_be_served_2['cluster'] = vendedor_df['Vendedor']
                    points_to_be_served_2['cluster'] = points_to_be_served_2['cluster'].astype("str")
                    points_to_be_served_2['check'] =vendedor_df['Cliente']

                    list_vendedor = list(vendedor_df['Vendedor'])

                    lat_ = []
                    long_ = []

                    for k in list_vendedor:
                        lat_.append(potential_locations['lat'][k])
                        long_.append(potential_locations['long'][k])

                    points_to_be_served_2['lat centro'] = lat_
                    points_to_be_served_2['long centro'] = long_

                    points_to_be_served_2['FTE Base Usado'] = fte_suporte +0.41
                    points_to_be_served_2['# P'] = p

                    layout = go.Layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        center=dict(lat=-15, lon=-56),
                        zoom=3.9
                    ),)

                    st.dataframe(points_to_be_served_2)

                    fig1 = px.scatter_mapbox(points_to_be_served_2, lat="lat", lon="long", title='Visão Final', height=400, width = 1250, color='cluster', hover_data=["Nome rede_ Ajustado", "Nome PDV_Ajustado"])
                    st.plotly_chart(fig1)

                else:
                     st.write(f'--> SEM SOLUÇÃO COM FTE = {fte_suporte+0.41}')



        with tab2:

            layout = go.Layout(
            mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=-15, lon=-55),
            zoom=3.2
            ),
            )

            fig_base_line = px.scatter_mapbox(df_base_clientes, lat="lat", lon="long",color =  'SubCoord_TOBE', title='Visualização de Clientes Alocados a Subcoords:',height = 600,width=1250, hover_data=["Nome PDV_Ajustado"])
            fig_base_line.update_layout(layout)
            st.plotly_chart(fig_base_line)

   
            subcoord_mapa = st.radio(
            "Subcoord:",
            options=list(subcoords_))


            if subcoord_mapa:

                DF_CLIENTS_SUBCOORD = df_base_clientes.query(f'SubCoord_TOBE =="{subcoord_mapa}"')
                DF_P = df_p.query(f'SubCoord_TOBE =="{subcoord_mapa}"')
                data = DF_CLIENTS_SUBCOORD[['lat', 'long']].values

                # Definir os limites de lat/long permitidos na região da base
                max_lat_base = max(data[:, 0])
                min_lat_base = min(data[:, 0])
                max_lon_base = max(data[:, 1])
                min_lon_base = min(data[:, 1])

                # 
                grid_points = generate_grid_points(max_lat_base, min_lat_base, max_lon_base, min_lon_base)
                grid_df = pd.DataFrame(grid_points, columns=['lat', 'long'])
                vetoed_points = veto_points(grid_points, data,distancia_de_veto)
                vetoed_grid_df = pd.DataFrame(vetoed_points, columns=['lat', 'long'])

                # Layout comum para os mapas
                layout = go.Layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        center=dict(lat=-15, lon=-56),
                        zoom=3.9
                    ),
                )

                # Crie os mapas separadamente
                fig3 = px.scatter_mapbox(grid_df, lat="lat", lon="long", title='Visão Grid:', height=600,width = 600)
                fig2 = px.scatter_mapbox(vetoed_grid_df, lat="lat", lon="long", title='Visão de Grid com Pontos de Veto:', height=600, width = 600)
     
                # Atualize o layout comum
                
                fig2.update_layout(layout)
                fig3.update_layout(layout)

                col3, col4 = st.columns([1,1])

                with col3:
                    st.plotly_chart(fig3)
                with col4:
                    st.plotly_chart(fig2)

    
    with tab3:

        st.write(" Esta ferramenta foi desenvolvida exclusivamente para a Dexco pela Pragmatis. Ela tem como objetivo otimizar a setorização de clientes,foi desenvolvida durante o projeto DEX04 - Implementação da Estrutura Comercial. \n  \n Todos os direitos são reservados à Pragmatis e a Dexco.")

        st.info("Em caso de dúvidas, contate: analytics@pragmatis.com.br")


            