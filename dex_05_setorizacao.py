import streamlit as st
import streamlit_nested_layout
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
from mip import Model, MINIMIZE, xsum, OptimizationStatus
from mip import Model, xsum, maximize, BINARY
import mip
import io
import base64

# teste 02

###
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
brazil_polygon = world[world.name == 'Brazil']['geometry'].values[0]


@st.cache_data 
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

@st.cache_data 
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


@st.cache_data 
def simulando (df_base_clientes,vetoed_grid_df,i,subcoords):
    
    st.write(subcoords)
    subcoords =  subcoords
    geral = pd.DataFrame({'lat':[],'long':[],'latitude_centro':[], 'longitude_centro':[], 'cluster_centro':[], 'SubCoord':[]})
    cont = 1
    sub = []
    status = []
    aux = 0
                    
    col1, col2, col3,col4 = st.columns(4)



    cont=cont +1
    DF_CLIENTS_SUBCOORD = df_base_clientes.query(f'SubCoord_TOBE =="{subcoords}"')
    df_p = DF_CLIENTS_SUBCOORD[['SubCoord_TOBE', 'p']].drop_duplicates()
    df_p['p'] = df_p['p'].astype(int)
    DF_P = df_p.query(f'SubCoord_TOBE =="{subcoords}"')
    data = DF_CLIENTS_SUBCOORD[['lat', 'long']].values




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

    c = distance_matrix

    p = list(DF_P['p'])[0]

    # # Definindo o problema:
    # col1.metric("Número de Potenciais Vendedores",len(potential_locations) )
    # col2.metric("Número de Clientes", len(points_to_be_served))
    # col3.metric("FTE Total de Atendimento", round(sum(FTE_Atendimento),2))
    # col4.metric("Número de Vendedores Máximo", p)

    num_tentativas = 0
    status_ =3

    FTE_max_aux = sorted(FTE_max).copy()
    # Adiciona FTEs para afrouxamento das restrições
    for i in range(4):
        FTE_max_aux.append(sorted(FTE_max)[-1])

    while ((status_ != OptimizationStatus.FEASIBLE and status_ != OptimizationStatus.OPTIMAL) and num_tentativas <= num_tentativas <= (len(FTE_max_aux))):

    #  print(f'Testando com FTE = {fte_suporte +0.41}...')

        if num_tentativas == len(FTE_max):
            print("Restrição de raio removida para obtenção da solução ótima")
        elif num_tentativas == len(FTE_max) + 1:
            print("Restrição de FTE mínimo removida para obtenção da solução ótima")
        elif num_tentativas == len(FTE_max) + 2:
            print("Restrição de FTE máximo removida para obtenção da solução ótima")
        elif num_tentativas == len(FTE_max) + 3:
            print("Restrição de breakeven removida para obtenção da solução ótima")


        vec_suporte = FTE_max_aux

        fte_suporte = vec_suporte[num_tentativas] - FTE_backoffice
        # Defining the problem:
        #prob = LpProblem("p-center", LpMinimize)
        prob = Model("p-center", sense=MINIMIZE)

        m = len(points_to_be_served)
        n = len(potential_locations)

        x = [[prob.add_var(var_type=BINARY) for j in range(n)] for i in range(m)]
        y = [prob.add_var(var_type=BINARY) for j in range(n)]

        # Define objective function
        prob.objective = xsum(c[i][j] * x[i][j] for i in range(m) for j in range(n))

        # Constraints:
        for i in range(m):
            prob += xsum(x[i][j] for j in range(n)) == 1

        for i in range(m):
            for j in range(n):
                prob += x[i][j] <= y[j]  # Restrição 1
                if num_tentativas < len(FTE_max): # Afrouxa a restrição do raio após a primeira iteração
                    prob += c[i][j] * x[i][j] <= x[i][j] * raio_max  # Restrição 2

        if p_menor_igual == 1:
            prob += xsum(y[j] for j in range(n)) <= p
        else:
            prob += xsum(y[j] for j in range(n)) == p


        
        for j in range(n):
            if num_tentativas < len(FTE_max) + 2:
                prob += xsum(x[i][j] * Visita[i] * (0.25 + (c[i][j] / 65)) for i in range(m)) / 220 + (xsum(x[i][j] * FTE_Atendimento[i] for i in range(m))) <= fte_suporte * y[j]

            if num_tentativas < len(FTE_max) + 1:
                prob += xsum(x[i][j] * Visita[i] * (0.25 + (c[i][j] / 65)) for i in range(m)) / 220 + (xsum(x[i][j] * FTE_Atendimento[i] for i in range(m))) >= (FTE_min - FTE_backoffice) * y[j]

            if num_tentativas < len(FTE_max) + 3:
                prob += xsum(x[i][j] * ROL[i] for i in range(m)) >= breakeven * y[j]


        for network, members in network_dict.items():
            network_members = [i for i, row in points_to_be_served_2.reset_index(drop=True).iterrows() if row['Nome rede_ Ajustado'] == network]
            if network_members:
                for nm, nms in zip(network_members[1:], network_members):
                    for j in range(n):
                        prob += x[nms][j] - x[nm][j] == 0

        tempo_inicial = time.time()

        if com_tempo_limite_de_processamente == 1:
            prob.optimize(max_seconds=tempo_limite)
        else:
            prob.optimize()

        # Verificando o status da solução
        status_ = prob.status
        print(status_)
        # Calculando o tempo decorrido
        tempo_decorrido = time.time() - tempo_inicial
        tempo_final = time.time()
        num_tentativas = num_tentativas+1


    if status_ == OptimizationStatus.OPTIMAL or status_ == OptimizationStatus.FEASIBLE:
        #st.success(f'Tem solução com FTE = {fte_suporte+0.41}, e o tempo de solução foi de {round(tempo_final-tempo_inicial,0)}s', icon="✅")
        # st.write(f'--> TEM SOLUÇÃO COM FTE = {fte_suporte+0.41} ')

        # st.write("--> tempo de solução =",round(tempo_final-tempo_inicial,0))

        st.write("\n")

        info_otim = []
        data_otim = {'Cluster': [],
                     'FTE Total': [],
                     'ROL Total': [],
                     '# de PDV': [],
                     'FTE - CM': [],
                     'Lat - CM': [],
                     'Lon - CM': []}
        
        for j in range(n):
            if y[j].x ==1:

                info_otim.append(f'CLUSTER - {j}')
                data_otim['Cluster'].append(j)

                info_otim.append(f'--> FTE Total: {round(sum(x[i][j].x * Visita[i] * (0.25 + (c[i][j] / 65)) for i in range(m)) / 220 + sum(x[i][j].x * FTE_Atendimento[i] for i in range(m)) + 0.41 ,2)}')
                data_otim['FTE Total'].append(round(sum(x[i][j].x * Visita[i] * (0.25 + (c[i][j] / 65)) for i in range(m)) / 220 + sum(x[i][j].x * FTE_Atendimento[i] for i in range(m)) + 0.41 ,2))

                info_otim.append(f'--> # de PDV:  {round(sum(x[i][j].x * 1 for i in range(m) ) ,0 )}' )
                data_otim['# de PDV'].append(round(sum(x[i][j].x * 1 for i in range(m) ) ,0 ))

                Lat_ = sum(x[i][j].x*ROL[i]*list(points_to_be_served_2['lat'])[i] for i in range(m)) /sum(ROL[i]*x[i][j].x for i in range(m))
                Lon_ = sum(x[i][j].x*ROL[i]*list(points_to_be_served_2['long'])[i] for i in range(m)) /sum(ROL[i]*x[i][j].x for i in range(m))

                fte_ = round(sum(x[i][j].x * Visita[i] * (0.25 + ( ( math.sqrt( (Lat_ -list(points_to_be_served_2['lat'])[i])**2 + ( Lon_ -list(points_to_be_served_2['long'])[i])**2)) / 65)) for i in range(m)) / 220 + sum(x[i][j].x*FTE_Atendimento[i] for i in range(m)) + 0.41 ,2)
    
                info_otim.append(f'--> FTE CM: {fte_} (Lat = {round(Lat_, 4)} Lon = {round(Lon_, 4)})')
                data_otim['FTE - CM'].append(fte_)
                data_otim['Lat - CM'].append(round(Lat_, 4))
                data_otim['Lon - CM'].append(round(Lon_, 4))

                info_otim.append(f'--> ROL Total: {round(sum(x[i][j].x * ROL[i] for i in range(m) ) ,0 )}')
                data_otim['ROL Total'].append(round(sum(x[i][j].x * ROL[i] for i in range(m) ) ,0 ))

                
        df_otim = pd.DataFrame(data_otim)
        lista_vendedor = []
        lista_cliente = []
        lista_fte_s_back = []
        
        for j in range(n):
            for i in range(m):
                if x[i][j].x:
                    lista_cliente.append(i)
                    lista_vendedor.append(j)

        for i in range(m):
            lista_fte_s_back.append(sum([x[i][j].x * Visita[i] * (0.25 + (c[i][j] / 65)) for j in range(n)]) / 220 + (sum([x[i][j].x * FTE_Atendimento[i] for j in range(n)])))


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
        
        df_otim["Cluster"] = df_otim["Cluster"].astype("str")
        df_otim = pd.merge(df_otim, points_to_be_served_2[['cluster', 'lat centro', 'long centro']], left_on='Cluster', right_on='cluster')
        df_otim = df_otim.rename(columns={'lat centro': 'Lat - Cluster', 'long centro': 'Lon - Cluster'})
        df_otim = df_otim[['Cluster', 'FTE Total', 'ROL Total', 'Lat - Cluster', 'Lon - Cluster', '# de PDV', 'FTE - CM', 'Lat - CM', 'Lon - CM']]
        df_otim = df_otim.drop_duplicates()

        points_to_be_served_2['FTE Base Usado'] = fte_suporte +0.41
        points_to_be_served_2['# P'] = p

        points_to_be_served_2['FTE Cliente - sem backoffice'] = lista_fte_s_back
        points_to_be_served_2['FTE Cliente - backoffice'] = 0.41 / points_to_be_served_2['cluster'].map(points_to_be_served_2['cluster'].value_counts())
        points_to_be_served_2['FTE Cliente - total'] = points_to_be_served_2['FTE Cliente - sem backoffice'] + points_to_be_served_2['FTE Cliente - backoffice']

    else:
        st.write(f'--> SEM SOLUÇÃO COM FTE = {fte_suporte+0.41}')

    return  points_to_be_served_2,info_otim,df_otim



@st.cache_data 
def visoes_grid_(df_base_clientes, subcoords_list):
    
    grid_ = []
    veto_grid = []

    df_metric = {'subcoord': [],
             'num_potenciais_vendedores':[],
             'num_clientes': [],
             'FTE_total': [],
             'num_vendedores_max': []}

    for subcoord_mapa in subcoords_list:

        

        DF_CLIENTS_SUBCOORD = df_base_clientes.query(f'SubCoord_TOBE =="{subcoord_mapa}"')
        
        data = DF_CLIENTS_SUBCOORD[['lat', 'long']].values

        # Definir os limites de lat/long permitidos na região da base
        max_lat_base = max(data[:, 0])
        min_lat_base = min(data[:, 0])
        max_lon_base = max(data[:, 1])
        min_lon_base = min(data[:, 1])


        grid_points = generate_grid_points(max_lat_base, min_lat_base, max_lon_base, min_lon_base)
        grid_df = pd.DataFrame(grid_points, columns=['lat', 'long'])
        vetoed_points = veto_points(grid_points, data,distancia_de_veto)
        vetoed_grid_df = pd.DataFrame(vetoed_points, columns=['lat', 'long'])

        grid_.append(grid_df)
        veto_grid.append(vetoed_grid_df)

        df_metric['subcoord'].append(subcoord_mapa)
        df_metric['num_potenciais_vendedores'].append(len(vetoed_grid_df))
        df_metric['num_clientes'].append(len(DF_CLIENTS_SUBCOORD))
        df_metric['FTE_total'].append(round(sum(DF_CLIENTS_SUBCOORD['FTE_Atend_Visita']),2))
        df_metric['num_vendedores_max'].append(list(DF_CLIENTS_SUBCOORD['p'])[0])
        

    df_metric = pd.DataFrame(df_metric)

    return grid_ , veto_grid, df_metric

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


st.markdown("""
            # GTM - Setorização de Clientes
            """)



tab1, tab2, tab3 = st.tabs(["Página Principal", "Visualização", "Sobre"])

with tab1:

    st.markdown("### Configuração:")
    st.write('Altere os parâmetros abaixo para realizar ajustes no simulador, em caso do modelo não encontrar solução para um determinado FTE desejado, recomendamos fortemente alterar o raio máximo de atendimento e o breakeven. Sempre que possível, verifique se não há nenhum ponto ponto outlier foram da região de análise desejada na aba "visualização".')
    with st.expander('', expanded=True):
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("**Logística:**")
            raio_max = st.number_input(
                "**Raio Máximo de Atendimento (km):**",
                value=700,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i")
            modelo_pontos_candidatos = ["Quadrantes com Veto", "Pontos de Clientes"]

            
            bg_shape = st.radio(
            "**Pontos Candidatos:**",
            options=modelo_pontos_candidatos,
            key="bg_shape")
            
            if bg_shape == "Quadrantes com Veto":
                modelo_grid = 1

            distancia_pontos_grid = st.number_input(
                "**Distancia entre Pontos (km):**",
                value=10,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i" )

            distancia_de_veto = st.number_input(
                "**Distancia de Veto (km):**",
                value=10,
                min_value=1,
                max_value=10000,
                step=1,
                format="%i" )
            
            bg_shape_options_2 = [ " == P"," <= P"]
            rd_restr = st.radio(
            "**Número de Vendedores:**",
            options=bg_shape_options_2)

            if rd_restr == " <= P":
                p_menor_igual = 1
                
            else:
                p_menor_igual = 0

            
        with col2:

            st.write("**Restrições:**")

            breakeven_mm = st.number_input(
                "**Breakeven (R$ MM)**",
                value=5.5,
                min_value=1.0,
                max_value=9.999999,
                step=0.1,
                format="%.1f"  # Formatação para exibir uma casa decimal
            )

            # Exibindo o valor em milhões
            breakeven = breakeven_mm * 1000000

            tempo_limite= st.number_input(
                "**Tempo limite de otimização:**",
                value=60,
                min_value=1,
                max_value=9999999,
                step=1)

            com_tempo_limite_de_processamente = 1

            st.write("**Full time equivalent (FTE):**")

            FTE_min = st.number_input(
                "**FTE Mínimo:**",
                value=0.60,
                min_value=0.0,
                max_value=2.0,
                step=0.05)

            FTE_backoffice = st.number_input(
                "**FTE Backoffice:**",
                value= 0.41,
                min_value=0.0,
                max_value=2.0,
                step=0.05)
            
            FTE_max = st.multiselect('**FTE Máximo:**',[0.95, 1, 1.05,1.10,1.15,1.20,1.25], [0.95, 1, 1.05,1.10,1.25])
            
    st.divider()
    st.markdown("### Entrada dos dados:")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Para utilizar o simulador, é necessário que a base contenha todas informações necessárias: nome da rede, nome do ponto de venda, latitude e longitude do ponto de venda, número de vendedores e a subcoord. Caso não possua a base padrão, clique no botão ao lado.")
    with col2:
        


        with open("DEX04 - FTE_Sede_NaoSede v4_Sem_DURA_DA_ENG.xlsx", "rb") as template_file:
            template_byte = template_file.read()

        excel_data = template_byte

        # Função para codificar o arquivo Excel em base64
        def get_base64_of_bin_file(bin_file):
            return base64.b64encode(bin_file).decode()

        # Codificar o arquivo Excel em base64
        b64_excel = get_base64_of_bin_file(excel_data)
        
        download_link = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}'

        # Estilo de botão visual
        st.markdown(
            f"""
            <div style="text-align: left;">
                <a href="{download_link}" download="base_padrao.xlsx" style="text-decoration: none;">
                    <div style="background-color: #858585; color: white; padding: 8px 20px; border-radius: 5px; display: inline-block;">
                        Download da base padrão
                    </div>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )



    base_de_clientes = st.file_uploader('Insira a Base de Clientes', type = ['xlsx'])

    if base_de_clientes:

        df_base_clientes = pd.read_excel(base_de_clientes)

        st.dataframe(df_base_clientes, height = 210, hide_index=True)

        lista_subcoords = list(set(list(df_base_clientes['SubCoord_TOBE'])))
        subcoords_ = st.multiselect('Subcoords',sorted(lista_subcoords),lista_subcoords[0:4])

        

        criar_visao_grid = st.checkbox("Utilizar essas subcoords:")
        st.info('Para visualizar o mapa de clientes por subcoord, acesse a aba "visualização" e selecione a opção acima para analisar o cenário AS IS por subcoord e os potenciais centros de massa dos vendedores TO BE.', icon="ℹ️")

        if criar_visao_grid:

            list_grid_ , list_veto, df_metric = visoes_grid_(df_base_clientes,subcoords_)

            st.divider()
            st.markdown("### Simulador:")
            st.write("""
                     Para simular o cenário TO BE das subcoords com a configuração mencionada acima, clique no botão "Simular Cenários TO BE". Antes de fazer isso, verifique no painel abaixo se é necessário qualquer alteração na entrada de dados.\n
                     Após a simulação de cada cenário, será possível baixar uma planilha com o resultado. Se desejar baixar uma versão com todas as simulações, aguarde a finalização de cada uma das subcoords.
                     """)

            st.write("##### Painel de subcoords: ")

            for subcoord in subcoords_:
                    
                st.write(f"**{subcoord}:**")

                col1, col2, col3,col4 = st.columns(4)
                
                # Definindo o problema:
                col1.metric("Número de Potenciais Vendedores",df_metric['num_potenciais_vendedores'][df_metric['subcoord'] == subcoord])
                col2.metric("Número de Clientes", df_metric['num_clientes'][df_metric['subcoord'] == subcoord])
                col3.metric("FTE Total de Atendimento", df_metric['FTE_total'][df_metric['subcoord'] == subcoord])
                col4.metric("Número de Vendedores Máximo", df_metric['num_vendedores_max'][df_metric['subcoord'] == subcoord])

                st.write("\n")

            simular = st.button("Simular Cenários TO BE")

            if simular:
                
                st.write("##### Resultados:")

                geral = pd.DataFrame({'lat':[],'long':[],'latitude_centro':[], 'longitude_centro':[], 'cluster_centro':[], 'SubCoord':[]})
                cont = 1
                sub = []
                status = []
                aux = 0

                for i in range(len(subcoords_)):
                    
                    subcoords = subcoords_[i]
                    with st.spinner(f"Simulando {subcoords_[i]}"):
                        df_cluster, info_otim, df_otim = simulando(df_base_clientes,list_veto[i],i,subcoords)

                    df_otim['ROL Total'] = df_otim['ROL Total'].apply(lambda x: f'R${x/1000000:.1f}MM')
                    df_otim[['Lat - Cluster', 'Lon - Cluster', 'Lat - CM', 'Lon - CM']] = df_otim[['Lat - Cluster', 'Lon - Cluster', 'Lat - CM', 'Lon - CM']].applymap(lambda x: round(x, 2))
                    # st.dataframe(df_otim, hide_index=True)

                    layout = go.Layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        center=dict(lat=-15, lon=-56),
                        zoom=3.9
                    ),
)
                    fig1 = px.scatter_mapbox(df_cluster, lat="lat", lon="long", height=400,width = 950, color='cluster', hover_data=["Nome rede_ Ajustado", "Nome PDV_Ajustado"])
                    fig1.update_layout(layout)
                    # st.plotly_chart(fig1)

                    # Botao de instalacao da planilha em Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_cluster.to_excel(writer, sheet_name='Sheet1')

                    output.seek(0)  # Volte para o início do arquivo antes de obter os bytes

                    excel_data = output.getvalue()

                    # Função para codificar o arquivo Excel em base64
                    def get_base64_of_bin_file(bin_file):
                        return base64.b64encode(bin_file).decode()

                    # Codificar o arquivo Excel em base64
                    b64_excel = get_base64_of_bin_file(excel_data)
                    
                    download_link = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}'

                    # # Estilo de botão visual
                    # st.markdown(
                    #     f"""
                    #     <div style="text-align: left;">
                    #         <a href="{download_link}" download="Resultado_da_setorizacao.xlsx" style="text-decoration: none;">
                    #             <div style="background-color: #4CAF50; color: white; padding: 8px 20px; border-radius: 5px; display: inline-block;">
                    #                 Download da tabela em Excel
                    #             </div>
                    #         </a>
                    #     </div>
                    #     """,
                    #     unsafe_allow_html=True
                    # )


                    # st.dataframe(df_cluster, height = 210, hide_index=True)
                    with st.expander(f"**{subcoords_[i]}**", expanded=False):
                        st.write("**Visão geral:**")
                        st.dataframe(df_otim, hide_index=True)

                        st.write("**Mapa por vendedores TO BE:**")
                        st.plotly_chart(fig1)

                        st.write("**Download da base:**")

                        st.write("Pré-visualização:")

                        st.dataframe(df_cluster, height = 210, hide_index=True)
                        # Estilo de botão visual
                        st.markdown(
                            f"""
                            <div style="text-align: left;">
                                <a href="{download_link}" download="DEX_GTM_{subcoords.replace("/", "-")}.xlsx" style="text-decoration: none;">
                                    <div style="background-color: #858585; color: white; padding: 8px 20px; border-radius: 5px; display: inline-block;">
                                        Download da tabela em Excel
                                    </div>
                                </a>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.write("\n")

                        

        with tab2:

            

            layout = go.Layout(
            mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=-15, lon=-55),
            zoom=3
            ),
            )

            
            st.write("### Mapa de clientes por subcoord: ")
            st.write("Observe no mapa abaixo a disposição de clientes por subcoord de acordo com a base de dados inserida, caso queira observar uma subcoord especifica, clique na legenda do mapa. Verifique com atenção caso haja algum cliente outlier presente dentro da base (i.e. com lat long errada).")

            fig_base_line = px.scatter_mapbox(df_base_clientes, lat="lat", lon="long",color =  'SubCoord_TOBE',height = 600,width=1050, hover_data=["Nome PDV_Ajustado"])
            fig_base_line.update_layout(layout)
            st.plotly_chart(fig_base_line)
            

            if criar_visao_grid:
                st.divider()
                st.write("### Visualização das subcoords a serem analisadas")
                st.write("Abaixo temos três visualizações disponíveis: (1) Mapa da subcoord com clientes por vendedores, (2) Visualização do grid de potenciais clientes e (3) Visualização dos potenciais centros de vendedores. Observe a visualização final dos potenciais centros de vendedores, caso ela cubra uma área maior do que esperada é sinal de que há outliers na base.\nSelecione a subcoord desejada:")

                subcoord_mapa = st.radio(
                "Subcoord:",
                options=list(subcoords_))

            else:
                subcoord_mapa = False

            if subcoord_mapa and criar_visao_grid:

                # Visualização dos clientes
                for subcoord in subcoords_:
                    df_para_viz_ASIS= df_base_clientes.query(f'SubCoord_TOBE =="{subcoord_mapa}"') #### TESTE123

                    layout = go.Layout(
                        mapbox=dict(
                            style="carto-darkmatter",
                            center=dict(lat=-15, lon=-55),
                            zoom=3.2
                        ),
                    )

                fig1_2 = px.scatter_mapbox(df_para_viz_ASIS, lat="lat", lon="long",color = "Base Pessoal AS IS' - Final",height = 600, width=1050, hover_data=["Nome PDV_Ajustado"])
                fig1_2.update_layout(layout)
                
                st.write("**Mapa por vendedores TO BE:**")
                st.plotly_chart(fig1_2)

                grid_df = list_grid_[subcoords_.index(subcoord_mapa)]
                vetoed_grid_df = list_veto[subcoords_.index(subcoord_mapa)]

                # Crie os mapas separadamente
                fig3 = px.scatter_mapbox(grid_df, lat="lat", lon="long", height=500,width = 550)
                fig2 = px.scatter_mapbox(vetoed_grid_df, lat="lat", lon="long", height=500, width = 550)
     
                # Atualize o layout comum
                
                fig2.update_layout(layout)
                fig3.update_layout(layout)

                with st.expander("Detalhamento - potenciais centros de vendedores:", expanded=False):
                    col3, col3_5, col4 = st.columns([5,1,5])

                    with col3:
                        st.write("**Grid dos potenciais centros de vendedores:**")
                        st.plotly_chart(fig3)
                    with col4:
                        st.write("**Potenciais centros de vendedores:**")
                        st.plotly_chart(fig2)

    
    with tab3:

        st.write(" Esta ferramenta foi desenvolvida exclusivamente para a Dexco pela Pragmatis. Ela tem como objetivo otimizar a setorização de clientes,foi desenvolvida durante o projeto DEX04 - Implementação da Estrutura Comercial. \n  \n Todos os direitos são reservados à Pragmatis e a Dexco.")

        st.info("Em caso de dúvidas, contate: analytics@pragmatis.com.br")


            