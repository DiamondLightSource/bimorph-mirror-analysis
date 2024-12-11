def find_voltages(data: np.ndarray, v: float, baseline_voltage_scan: str = 'first' ) -> np.ndarray:
    '''
    given a matrix of beamline centroid data, with columns of beamline scans at different actuator voltages and rows of slit positions, calculate the necessary voltages corrections to achive the target centroid position.

    Parameters:
    data: np.ndarray
        matrix of beamline centroid data, with rows of different slit positions and columns of pencil beam scans at different actuator voltages
    v: float
        The voltage increment applied to the actuators between pencil beam scans
    baseline_voltage_scan: str
        The pencil beam scan to use as the baseline for the centroid calculation. Options are 'first' and 'last'. 
        'first' uses the first pencil beam scan, where no incremental voltage has been applied to the actuators. 
        'last' uses the last pencil beam scan, where the incremental voltage has been applied to every actuator.

    Returns:
    np.ndarray
        array of voltage corrections required to move the centroid of each pencil beam scan to the target position
    '''
    responses = np.diff(data, axis=1) # calculate the response of each actuator by subtracting previous pencil beam 

    H = responses / v #response per unit charge
    H = np.hstack((np.ones((H.shape[0],1)),H)) #add columns of 1's to the left of H
    H_inv = np.linalg.pinv(H) # calculate the Moore-Penrose pseudo inverse of H


    if baseline_voltage_scan == 'first':
        baseline_voltage_beamline_positions = data[:,0]
    elif baseline_voltage_scan == 'last':
        baseline_voltage_beamline_positions = data[:,-1]
    else:
        raise ValueError('baseline_voltage_scan must be either "first" or "last"')

    target = np.mean(baseline_voltage_beamline_positions)
    Y = target - baseline_voltage_beamline_positions
        
    voltage_corrections = np.matmul(H_inv,Y) #calculate the voltage required to move the centroid to the target position

    return voltage_corrections[1:] #return the voltages required to move the centroid of each pencil beam scan to the target position