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
    maxTime = 60#10800
    
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
    mapK = {}
    for i in range(len(Vd)):
        mapK[K[i]] = Vd[i][0]
    totalNodes = len(Vc)+len(Vf)+len(Vd)
    

    #valor grande
    BM = 999999999


    #creamos modelo
    problem = pulp.LpProblem('MDVRP', pulp.LpMinimize)
    #creamos variables
    x = pulp.LpVariable.dicts('x', ((i,j,k) for i in range(totalNodes) for j in range(totalNodes) for k in range(len(K))), lowBound = 0, upBound = 1, cat = 'Binary')
    y = pulp.LpVariable.dicts('y', ((i,j,k) for i in range(totalNodes) for j in range(totalNodes) for k in range(len(K))), lowBound = 0, cat = 'Continuous')
    z = pulp.LpVariable.dicts('y', ((i,k) for i in range(totalNodes) for k in range(len(K))), lowBound = 0, upBound = 1, cat = 'Binary')
    #funcion objetivo
    problem += 0.5 * pulp.lpSum(dist[i][j]*x[i,j,k] for i in range(totalNodes) for j in range(totalNodes) for k in range(len(K)))
    #restricciones
    #(2)
    for i in Vc:
        for k in K:
            problem += pulp.lpSum(y[j,i[0],k]-y[i[0],j,k] for j in range(totalNodes) if i[0]!=j) == 2*i[4]*z[i[0],k]

    #(3)
    problem += pulp.lpSum(y[Vd[i][0],Vc[j][0],k] for i in range(len(Vd)) for j in range(len(Vc)) for k in range(len(K))) == pulp.lpSum(Vc[j][4] for j in range(len(Vc)))
    #(4)
    problem += pulp.lpSum(y[Vc[j][0],Vd[i][0],k] for i in range(len(Vd)) for j in range(len(Vc)) for k in range(len(K))) <= (pulp.lpSum(vehicleLoad[k] for k in range(len(K))) - pulp.lpSum(Vc[j][4] for j in range(len(Vc))) )
    #(5)
    for i in Vf:
        for k in K:
            problem += pulp.lpSum(y[i[0],Vc[j][0],k] for j in range(len(Vc)) if i[0] != j) <= vehicleLoad[k]
    #(6)
    for j in Vc:
        for k in K:
            problem += pulp.lpSum(x[i,j[0],k] for i in range(totalNodes) if i!= j[0]) == 2*z[j[0],k]
    #(7)
    for i in range(totalNodes):
        for j in range(totalNodes):
            for k in range(len(K)):
                problem += y[i,j,k]+y[j,i,k] == vehicleLoad[k]*x[i,j,k]
    #(8)
    for i in Vc:
        problem += pulp.lpSum(z[i[0],k] for k in range(len(K)))

    #(9)
    for i in Vc:
        for j in range(totalNodes):
            for k in K:
                if i[0]!=j:
                    problem += y[i[0],j,k] <= BM*z[i[0],k]

    #(10)
    for k in K:
        problem += pulp.lpSum(Vc[i][3]*x[Vc[i][0],j,k] for i in range(len(Vc)) for j in range(totalNodes) if Vc[i][0]!=j) + pulp.lpSum(dist[Vc[i][0]][j]*x[Vc[i][0],j,k] for i in range(len(Vc)) for j in range(totalNodes) if Vc[i]!=j) <= 2*depotDuration[0]
    #(11) Tenemos un solo vehiculo por depot por lo que podemos utilizar i tanto en i como en k
    for i in Vd:
        for k in K:
            if mapK[k]==i[0]:
                problem += pulp.lpSum(x[i[0],Vc[j][0],k] for j in range(len(Vc))) <= 1
    
    #(12)
    for j in Vf:
        for k in K:
            if j[0] != mapK[k]:
                problem += pulp.lpSum(x[Vc[i][0],j[0],k] for i in range(len(Vc)) if Vc[i][0]!=j[0]) == 0
    #(13)
    for i in Vd:
        for k in K:
            if mapK[k] != i[0]:
                problem += pulp.lpSum(x[i[0],Vc[j][0],k] for j in range(len(Vc)) if Vc[j][0]!=i[0]) == 0
   
       
    
    solver = pulp.GUROBI_CMD(timeLimit= 300)  
    
    # solve problem
    status = problem.solve()

    # output status, value of objective function
    status, pulp.LpStatus[status], pulp.value(problem.objective)
        

if __name__ == "__main__":
    args = sys.argv
    main(args)