def find_voltages(data: np.ndarray, v: float, baseline_voltage_scan: int = 0 ) -> np.ndarray:
    '''
    given a matrix of beamline centroid data, with columns of beamline scans at different actuator voltages and rows of slit positions, calculate the necessary voltages corrections to achive the target centroid position.

    Parameters:
    data: np.ndarray
        matrix of beamline centroid data, with rows of different slit positions and columns of pencil beam scans at different actuator voltages
    v: float
        The voltage increment applied to the actuators between pencil beam scans
    baseline_voltage_scan: int
        The pencil beam scan to use as the baseline for the centroid calculation. 0 is the first scan, 1 is the second scan, etc.
        -1 can be used for the last scan and -2 for the second to last scan etc.

    Returns:
    np.ndarray
        array of voltage corrections required to move the centroid of each pencil beam scan to the target position
    '''

    if not isinstance(baseline_voltage_scan, int):
        raise TypeError('baseline_voltage_scan must be an integer')
    if baseline_voltage_scan < -data.shape[1] or baseline_voltage_scan >= data.shape[1]:
        raise IndexError(f'baseline_voltage_scan is out of range, it must be between {-1*data.shape[1]} and {data.shape[1]-1}')
    

    responses = np.diff(data, axis=1) # calculate the response of each actuator by subtracting previous pencil beam 

    H = responses / v #response per unit charge
    H = np.hstack((np.ones((H.shape[0],1)),H)) #add columns of 1's to the left of H
    H_inv = np.linalg.pinv(H) # calculate the Moore-Penrose pseudo inverse of H

    baseline_voltage_beamline_positions = data[:,baseline_voltage_scan]


    target = np.mean(baseline_voltage_beamline_positions)
    Y = target - baseline_voltage_beamline_positions
        
    voltage_corrections = np.matmul(H_inv,Y) #calculate the voltage required to move the centroid to the target position

    return voltage_corrections[1:] #return the voltages required to move the centroid of each pencil beam scan to the target position