import numpy as np
import pandas as pd
import orbital_conversions
import planetary_body_config
import csv

class OutputDataMCI:
    def __init__(self, csvPath):
        data = pd.read_csv(csvPath, sep=',')

        self.time = np.asarray(data['time'])
        self.mass = np.asarray(data['mass'])
      
        self.r_mci = []
        self.r_ned = []
        self.v_mci = []
        self.thrust_vector_mci = []
        self.thrust_vector_ned = []
        self.altitude = []
        self.v_mci_mag = []

        r_mci_x = np.asarray(data['r_mci_x'])
        r_mci_y = np.asarray(data['r_mci_y'])
        r_mci_z = np.asarray(data['r_mci_z'])
        v_mci_x = np.asarray(data['v_mci_x'])
        v_mci_y = np.asarray(data['v_mci_y'])
        v_mci_z = np.asarray(data['v_mci_z'])
        thrust_vector_mci_x = np.asarray(data['thrust_mci_x'])
        thrust_vector_mci_y = np.asarray(data['thrust_mci_y'])
        thrust_vector_mci_z = np.asarray(data['thrust_mci_z'])
        for i in range(len(r_mci_x)):
           self.r_mci.append(np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]]))
           print(orbital_conversions.MCI_to_NED(np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]])) - orbital_conversions.MCI_to_NED2(np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]]), np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]])))
           self.r_ned.append(orbital_conversions.MCI_to_NED(np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]])))
           self.altitude.append(np.linalg.norm(np.array([r_mci_x[i], r_mci_y[i], r_mci_z[i]])) - planetary_body_config.body_radius)
           self.v_mci.append(np.array([v_mci_x[i], v_mci_y[i], v_mci_z[i]]))
           self.v_mci_mag.append(np.linalg.norm(np.array([v_mci_x[i], v_mci_y[i], v_mci_z[i]])))
           self.thrust_vector_mci.append(np.array([thrust_vector_mci_x[i], thrust_vector_mci_y[i], thrust_vector_mci_z[i]]))
           self.thrust_vector_ned.append(orbital_conversions.MCI_to_NED(np.array([thrust_vector_mci_x[i], thrust_vector_mci_y[i], thrust_vector_mci_z[i]])))
        
        self.r_mci = np.asarray(self.r_mci)
        self.r_ned = np.asarray(self.r_ned)
        self.altitude = np.asarray(self.altitude)
        self.v_mci = np.asarray(self.v_mci)
        self.v_mci_mag = np.asarray(self.v_mci_mag)
        self.thrust_vector_mci = np.asarray(self.thrust_vector_mci)
        self.thrust_vector_ned = np.asarray(self.thrust_vector_ned)
        

class OutputDataXYZ:
    def __init__(self, csvPath):
        data = pd.read_csv(csvPath, sep=',')

        self.time = np.asarray(data['time'])
        self.mass = np.asarray(data['mass'])
      
        self.r_xyz = []
        self.v_xyz = []
        self.thrust_vector_xyz = []
        self.altitude = []
        self.v_xyz_mag = []
        self.thrust_mag = []

        r_xyz_x = np.asarray(data['r_xyz_x'])
        r_xyz_y = np.asarray(data['r_xyz_y'])
        r_xyz_z = np.asarray(data['r_xyz_z'])
        v_xyz_x = np.asarray(data['v_xyz_x'])
        v_xyz_y = np.asarray(data['v_xyz_y'])
        v_xyz_z = np.asarray(data['v_xyz_z'])
        thrust_vector_xyz_x = np.asarray(data['thrust_xyz_x'])
        thrust_vector_xyz_y = np.asarray(data['thrust_xyz_y'])
        thrust_vector_xyz_z = np.asarray(data['thrust_xyz_z'])

        for i in range(len(r_xyz_x)):
           self.r_xyz.append(np.array([r_xyz_x[i], r_xyz_y[i], r_xyz_z[i]]))
           self.altitude.append(r_xyz_x[i])
           self.v_xyz.append(np.array([v_xyz_x[i], v_xyz_y[i], v_xyz_z[i]]))
           self.v_xyz_mag.append(np.linalg.norm(np.array([v_xyz_x[i], v_xyz_y[i], v_xyz_z[i]])))
           self.thrust_vector_xyz.append(np.array([thrust_vector_xyz_x[i], thrust_vector_xyz_y[i], thrust_vector_xyz_z[i]]))
           self.thrust_mag.append(np.linalg.norm(np.array([thrust_vector_xyz_x[i], thrust_vector_xyz_y[i], thrust_vector_xyz_z[i]])))

        self.r_xyz = np.asarray(self.r_xyz)
        self.altitude = np.asarray(self.altitude)
        self.v_xyz = np.asarray(self.v_xyz)
        self.v_xyz_mag = np.asarray(self.v_xyz_mag)
        self.thrust_vector_xyz = np.asarray(self.thrust_vector_xyz)
        self.thrust_mag = np.asarray(self.thrust_mag)
        



def logDataMCI(data):
  dataArray = [['time', 
                'mass', 
                'r_mci_x', 
                'r_mci_y', 
                'r_mci_z', 
                'v_mci_x', 
                'v_mci_y', 
                'v_mci_z', 
                'thrust_mci_x', 
                'thrust_mci_y', 
                'thrust_mci_z']]
  
  for i in range(len(data[0])):
    dataRow = []
    for j in range(len(data)):
      dataRow.append(data[j][i])
    dataArray.append(dataRow)   
  with open('outputs/standalone/data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(dataArray)

def logDataxyz(data):
  dataArray = [['time', 
                'mass', 
                'r_xyz_x', 
                'r_xyz_y', 
                'r_xyz_z', 
                'v_xyz_x', 
                'v_xyz_y', 
                'v_xyz_z', 
                'thrust_xyz_x', 
                'thrust_xyz_y', 
                'thrust_xyz_z']]
  
  for i in range(len(data[0])):
    dataRow = []
    for j in range(len(data)):
      dataRow.append(data[j][i])
    dataArray.append(dataRow)   
  with open('outputs/standalone/data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(dataArray)