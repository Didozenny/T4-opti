#import itertools

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import pulp
import sys

import warnings
warnings.filterwarnings("ignore")

def MatrizDistancia(clientData,depotData):
    xvar = []
    yvar = []
    #agregamos coordenadas de costumers
    for i in range (len(clientData)):
        xvar.append(clientData[i][1])
        yvar.append(clientData[i][2])
    #agregamos coordenadas de depots
    for i in range (len(depotData)):
        xvar.append(depotData[i][1])
        yvar.append(depotData[i][2])
    #agregamos copias de los depots
    for i in range (len(depotData)):
        xvar.append(depotData[i][1])
        yvar.append(depotData[i][2])
    #creamos un dataframe para calcular distancias
    df = pd.DataFrame({
        'x': xvar,
        'y': yvar,
    })
    #calculamos distancias
    distances = pd.DataFrame(distance_matrix(df[['x', 'y']].values,df[['x', 'y']].values), index = df.index, columns = df.index).values
    return distances
 
def main(args):
    #obtenemos nombre de archivo y agregamos nodos si corresponde
    fileN = args[1]
    numNodes = 25
    if(len(args) == 3 and args[3] == "moreNodes"):
        numNodes = 40
    maxTime = 10800
    
    file = open(fileN)
    #primera linea
    line = file.readline()
    #asumimos que el primer valor es de 2 por el alcance de la tarea y obtenemos el resto de argumentos
    nVehicles = int(line.split(" ")[1])
    nCostumers = int(line.split(" ")[2])
    nDepots = int(line.split(" ")[3])

    vehicleLoad = []
    depotDuration = []
    clientData = []
    depotData = []
    #obtenemos load y duration de los depositos
    for i in range(nDepots):
        line = file.readline()
        depotDuration.append(int(line.split(" ")[0]))
        vehicleLoad.append(int(line.split(" ")[1]))
    
    # se obtienen los datos de los clientes 
    # i x y d(service duration) q(demand) f(frequency of visit) a(combinaciones posibles) list(lista de posibles combinaciones)
    #almacenamos estos valores en clientData
    for i in range(numNodes):
        line = file.readline()
        data = line.split(" ")
        data = [x for x in data if x!= '' and x != '\n']
        data = [x.strip('\n') for x in data]
        clientData.append(data)
    
    #saltamos lineas hasta llegar a el ultimo costumer
    line = file.readline()
    data = line.split(" ")
    data = [x for x in data if x!= '' and x != '\n']
    while(int(data[0]) < nCostumers):
        line = file.readline()
        data = line.split(" ")
        data = [x for x in data if x!= '' and x != '\n']
    #obtenemos la data de los depots, en caso de que 
    for i in range(nDepots):
        if(int(data[0]) > nCostumers and i==0):
            data = [x.strip('\n') for x in data]
            depotData.append(data)
            continue
        line = file.readline()
        data = line.split(" ")
        data = [x for x in data if x!= '' and x != '\n']
        data = [x.strip('\n') for x in data]
        depotData.append(data)
    #cerramos archivo
    file.close()

    #pasamos la data obtenida a valores int y float
    auxClientData = []
    auxDepotData = []
    for cData in clientData:
        aux = [int(x) if '.' not in x else float(x) for x in cData ]
        auxClientData.append(aux)
    for dData in depotData:
        aux = [int(x) if '.' not in x else float(x) for x in dData ]
        aux[0] = aux[0] - nCostumers + numNodes #cambiamos el valor del indice para calzar con el siguiente 
        auxDepotData.append(aux)
    clientData = auxClientData
    depotData = auxDepotData
    #obtenemos matriz de distancia
    dist = MatrizDistancia(clientData,depotData)

    #comenzamos el two comodity flow formulation
    #creamos los sets para poder utilizarlos dentro del programa
    Vc = clientData
    Vd = depotData
    #copia de depot
    Vf = []
    for i in range(len(depotData)):
        Vf.append(depotData[i].copy())
    #hay que cambiar los indices para que coincidan con los indices de las distancias asi
    for i in range(len(Vc)):
        Vc[i][0] = Vc[i][0]-1
    for i in range(len(Vd)):
        Vd[i][0] = Vd[i][0]-1
    for i in range (len(Vf)):
        Vf[i][0] = Vf[i][0]+3
    K = list(range(0,nDepots))
    totalNodes = (Vc)+(Vd)+(Vf)
    mapK = {}
    for i in range(len(Vd)):
        mapK[K[i]] = Vd[i][0]

    # CREAR VARIABLES Y FUNCION OBJETIVO
    problem = pulp.LpProblem('mdvrp', pulp.LpMinimize)

    # Variables de decisión
    x = pulp.LpVariable.dicts('x', ((i, j, k) for i in range(len(totalNodes)) for j in range(len(totalNodes)) for k in range(len(K))), lowBound = 0, upBound = 1, cat = 'Binary')
    #d = pulp.LpVariable.dicts('d', ((i, j) for i in range(len(totalNodes)) for j in range(len(totalNodes))), lowBound = 0 , cat = 'Continuous')
    u = pulp.LpVariable.dicts('u', ((i) for i in range(len(totalNodes))), lowBound = 0 , cat = 'Continuous') 
    #funcion objetivo
    problem += pulp.lpSum(x[i,j,k] * dist[i][j] for i in range(len(totalNodes)) for j in range(len(totalNodes)) for k in range(len(K)))


    # RESTRICCIONES

    # RESTRICCION (16)
    for j in Vc:
        problem += pulp.lpSum([x[(i[0],j[0],k)] for i in totalNodes for k in K if i[0]!=j[0]]) == 1

    # RESTRICCION (17)
    for i in Vc:
        problem += pulp.lpSum([x[(i[0],j[0],k)] for j in totalNodes for k in K if i[0]!=j]) == 1

    # RESTRICCION (18)
    for h in totalNodes:
        for k in K:
            problem += (pulp.lpSum([x[(i[0],h[0],k)] for i in totalNodes if i[0]!=h]) - pulp.lpSum([x[(h[0],j[0],k)] for j in totalNodes if j[0]!=h[0]])) == 0

    # RESTRICCION (19)
    for k in K:
            problem += pulp.lpSum(i[4] * x[i[0],j[0],k] for i in Vc for j in totalNodes) <= vehicleLoad[k] 

    # RESTRICCION (20)          ti service duration at customer i        rij travelling time from node i to node j  T maximum time allowed for a route
    for k in K:
            problem += (pulp.lpSum([i[3] * x[i[0],j[0],k] for i in totalNodes for j in totalNodes]) + pulp.lpSum([dist[i[0],j[0]] * x[i[0],j[0],k] for i in totalNodes for j in totalNodes])) <=   depotDuration[0]

    # RESTRICCION (21)
    for i in Vd:
        for k in K:
            if mapK[k]==i[0]:
                problem += pulp.lpSum(x[i[0],j[0],k] for j in Vc) <= 1 

    # RESTRICCION (22)
    for j in Vd:
        for k in K:
            if mapK[k]==j[0]:
                problem += pulp.lpSum(x[i[0],j[0],k] for j in Vc) <= 1 


    # RESTRICCION (23)
    for k in K:
        for j in Vd:
            if j[0] != mapK[k]:
                problem += pulp.lpSum(x[i[0],j[0],k] for j in Vc) == 0
    
    # RESTRICCION (24)
    for k in K:
        for i in Vd:
            if i[0] != mapK[k]:
                problem += pulp.lpSum(x[i[0],j[0],k] for j in Vc) == 0 



    # RESTRICCION (25) ----------------- MTZ

    for k in K:
        for i in Vc:
            for j in Vc:
                        if i[0] != j[0]:
                            problem += u[i[0]] - u[j[0]] + len(Vc) * x[i[0],j[0],k] <= len(Vc) - 1
    



# solve problem
    solver = pulp.GUROBI_CMD(timeLimit= 10800)
    status = problem.solve(solver)
    
    # output status, value of objective function
    status, pulp.LpStatus[status], pulp.value(problem.objective)
        

if __name__ == "__main__":
    args = sys.argv
    main(args)
