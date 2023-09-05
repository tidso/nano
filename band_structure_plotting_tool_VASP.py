#IMPORT
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import gridspec
import seaborn as sns
from matplotlib import style
import plotly.graph_objs as go
import plotly.io as pio

#FUNCTIONS/DEFINITIONS
yes = {'yes','y', 'ye', ''}
no = {'no','n'} 

def yes_or_no(question):
    while True:
        reply = str(input(question + ' (yes/no): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        elif reply[:1] == 'n':
            return False
        else:
            print("Invalid input. Please answer with 'yes' or 'no'.")

#Read the data file, .txt with 5 columns (Energy, DOS UP, and DOS DOWN are the first three columns)
def read_doscar_data(DOSCAR_VASP_FILE):
    ENERGY = []
    DOS_U = []
    DOS_D = []
    #INT_DOS_U = []
    #INT_DOS_D = []
    with open(DOSCAR_VASP_FILE) as datafile:
        for i in range(5):
            datafile, next(datafile)
        how_many_lines = datafile.readline()
        how_many_lines = how_many_lines.strip()
        line_elements = re.split(" +", how_many_lines)
        line_count = int(line_elements[2])
        j=0
        for line in datafile:
            if j < line_count:
                line=line.strip()
                values = line.split()
                ENERGY.append(values[0])
                DOS_U.append(values[1])
                DOS_D.append(values[2])
                #INT_DOS_U.append(values[3])
                #INT_DOS_D.append(values[4])
                j += 1
            elif j >= line_count:
                pass
    ENERGY = np.array(ENERGY, dtype = 'float')
    DOS_U = np.array(DOS_U, dtype = 'float')
    DOS_D = np.array(DOS_D, dtype = 'float')
    #INT_DOS_U = np.array(INT_DOS_U, dtype = 'float')
    #INT_DOS_D = np.array(INT_DOS_U, dtype = 'float')
    return ENERGY, DOS_U, DOS_D #, INT_DOS_U, INT_DOS_D

def dos_plotting_tool(DOSCAR_path):

    #Obtain arrays of data from the datafile
    ENERGY = read_doscar_data(DOSCAR_path)[0]
    DOS_U = read_doscar_data(DOSCAR_path)[1]
    DOS_D = read_doscar_data(DOSCAR_path)[2]
    E_Fermi = fermi_energy(DOSCAR_path)

    ENERGY = np.array(ENERGY)
    ENERGY_FERMI_SHIFTED = np.array(ENERGY - E_Fermi)
    xmin = np.min(ENERGY_FERMI_SHIFTED) - 1
    xmax = np.max(ENERGY_FERMI_SHIFTED) + 1
    DOS_U = np.array(DOS_U)
    DOS_D = -1 * np.array(DOS_D)

    #INT_DOS_U = np.array(read_data(filename)[3])
    #INT_DOS_D = np.array(read_data(filename)[4])
    #print(f'The total number of UP states is: {np.sum(INT_DOS_U)}')
    #print(f'The total number of DOWN states is: {np.sum(INT_DOS_D)}')


    #Stylize the Graph
    plt.rcParams.update({'font.family':'serif'})
    plt.style.use('ggplot')
    plt.figure(layout='tight')
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('font', size=10)          # controls default text sizes

    plt.plot(ENERGY_FERMI_SHIFTED,DOS_U,color=sns.color_palette()[1],label='Spin-Up')
    plt.plot(ENERGY_FERMI_SHIFTED,DOS_D,color=sns.color_palette()[0],label='Spin-Down')
    plt.fill_between(ENERGY_FERMI_SHIFTED,np.zeros_like(DOS_U),DOS_U, where = (ENERGY_FERMI_SHIFTED < 0), color=sns.color_palette()[1],alpha=0.4)
    plt.fill_between(ENERGY_FERMI_SHIFTED,DOS_D,np.zeros_like(DOS_D), where = (ENERGY_FERMI_SHIFTED < 0), color=sns.color_palette()[0],alpha=0.4)
    plt.axvline(x=0, label='Fermi Level', alpha=0.85, color = 'gray', ls='--')

    # XMIN,XMAX IF NECESSARY
    plt.xlim([xmin,xmax])

    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV/cell)')
    plt.legend()

    plt.title(f'DOS of {SYSTEM_NAME}')
    if yes_or_no("Would you like to save the DOS plot? "):
        plt.savefig(f'{SYSTEM_NAME}_dos.pdf')

def interactive_dos_plotting_tool(DOSCAR_path, SYSTEM_NAME):
    #Obtain arrays of data from the datafile
    ENERGY = read_doscar_data(DOSCAR_path)[0]
    DOS_U = read_doscar_data(DOSCAR_path)[1]
    DOS_D = read_doscar_data(DOSCAR_path)[2]
    E_Fermi = fermi_energy(DOSCAR_path)

    ENERGY = np.array(ENERGY)
    ENERGY_FERMI_SHIFTED = np.array(ENERGY - E_Fermi)
    xmin = np.min(ENERGY_FERMI_SHIFTED) - 1
    xmax = np.max(ENERGY_FERMI_SHIFTED) + 1
    DOS_U = np.array(DOS_U)
    DOS_D = -1 * np.array(DOS_D)

    # Create the interactive plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ENERGY_FERMI_SHIFTED, y=DOS_U,
                             mode='lines', name='Spin-Up',
                             line=dict(color='red')))

    fig.add_trace(go.Scatter(x=ENERGY_FERMI_SHIFTED, y=DOS_D,
                             mode='lines', name='Spin-Down',
                             line=dict(color='blue')))

    fig.update_layout(title=f'Density of States of {SYSTEM_NAME}',
                      xaxis_title='Energy (eV)',
                      yaxis_title='DOS (states/eV/cell)',
                      xaxis=dict(range=[xmin, xmax]),
                      showlegend=True,
                      hovermode='closest')

    fig.add_shape(type='line',
                  x0=0, x1=0,
                  y0=np.min(DOS_D), y1=np.max(DOS_U),
                  yref='y',
                  xref='x',
                  line=dict(color='gray', dash='dash'))

    # Save the plot as an html file
    if yes_or_no("Would you like to save the interactive DOS plot? "):
        pio.write_html(fig, file=f'{SYSTEM_NAME}_interactive_dos.html', auto_open=True)
    else:
        fig.show()

#Determines the magnitude of the distance between two kpoints
def vect_magnitude(x1,x2,y1,y2,z1,z2):
    return np.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

#Read the data file, Use KPOINTS (VASP input, linemode only) as txt file
def read_kpoints_file(filename):
    coordinates = []
    path = []
    with open(filename) as f:
        for i in range(4):
            f, next(f)
        for line in f:
            if line != '\n':
                line=line.strip()
                contents = re.split(" +", line)
                x = float(contents[0])
                y = float(contents[1])
                z = float(contents[2])
                k = str(contents[-1])
                coordinates.append((x,y,z))
                path.append(k)
        coordinates = np.array(coordinates)
        
        i = 0
        new_path = []
        while i <= len(path) - 2:
            new_path.append(str(path[i] + '→' + path[i+1]))
            i += 1
        del new_path[1::2]
        new_path.append('states/eV')
        
        j = 0
        path_magnitudes = []
        while j < len(path) - 1:
            path_magnitudes.append(vect_magnitude(coordinates[j][0], coordinates[j+1][0], coordinates[j][1], coordinates[j+1][1], coordinates[j][2], coordinates[j+1][2]))
            j += 1
        del path_magnitudes[1::2]
        
    return path_magnitudes, new_path

#Normalizes the paths so that the graph can be plotted for relative path size on the x-axis
def normalized_path_mag(filename):
    raw = read_kpoints_file(filename)[0]
    norm = [float(i)/sum(raw) for i in raw]
    norm.append(float(0.25))
    return norm
    
#Reads EIGENVAL to determine the number of eigenvalues that need to be plotted
def number_of_electrons_kpoints_bands(filename2):
    with open(filename2) as f:
        for i in range(5):
            f, next(f)
        my_line = f.readline()
        my_line = my_line.strip()
        values = re.split(" +", my_line)
        num_of_electrons = int(values[0])
        num_of_kpoints = int(values[1])
        num_of_bands = int(values[2])
    return num_of_electrons, num_of_kpoints, num_of_bands

#Read the data file, Use KPOINTS (VASP input, linemode only) as txt file
def subdivisions(filename):
    with open(filename) as f:
        f.readline()
        line2 = f.readline()
        line2 = line2.strip()
        vals = re.split(" +", line2)
        subdivisions = int(vals[0])
    return subdivisions

#Determines the number of paths that will need to be plotted for Band Structure
def number_of_paths(filename, filename2):
    number_of_kpoints = number_of_electrons_kpoints_bands(filename2)[1]
    subdivisions_num = subdivisions(filename)
    return int(number_of_kpoints / subdivisions_num)

def obtain_KPOINTS_paths(filename2, subdivisions_per_path):
    KPOINTS = []
    repeating_factor = 2 + number_of_electrons_kpoints_bands(filename2)[2]
    
    with open(filename2) as datafile:
        for i in range(7):
            datafile, next(datafile)
        for linenum, line in enumerate(datafile, start=1):
            if linenum % (repeating_factor) == 1:
                line=line.strip()
                values = re.split(" +", line)
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                subdivision = np.sqrt(x**2 + y**2 + z**2)
                KPOINTS.append(subdivision)
        # How many elements each
        # list should have
        n = subdivisions_per_path

        # using list comprehension
        KPOINTS_BY_PATH = [KPOINTS[i * n:(i + 1) * n] for i in range((len(KPOINTS) + n - 1) // n )]
        KPOINTS_BY_PATH = np.array(KPOINTS_BY_PATH)
        KPOINTS_BY_PATH.sort(axis=1)
    return KPOINTS_BY_PATH
           
#Read the data file, Use EIGENVAL (VASP output) as txt file
def obtain_EVALS(filename2, subdivisions_per_path):
    EVALS = []
    repeating_factor = 2 + number_of_electrons_kpoints_bands(filename2)[2]
        
    with open(filename2) as datafile:
        for i in range(7):
            datafile, next(datafile)
        for linenum, line in enumerate(datafile, start=1):
            if linenum % repeating_factor != 1:
                if linenum % repeating_factor != 0:
                    line=line.strip()
                    values = line.split()
                    band_number = int(values[0])
                    energy_up = float(values[1])
                    energy_down = float(values[2])
                    
                    EVALS.append([band_number,energy_up,energy_down])
        # How many elements each
        # list should have
        n = subdivisions_per_path * (repeating_factor - 2)

        # using list comprehension
        eigenvalues_by_path = [EVALS[i * n:(i + 1) * n] for i in range((len(EVALS) + n - 1) // n )]
        
    return eigenvalues_by_path

#Sort the eigenvalues such that they are ready to plot
def evals_sort(eigenvalues_by_path, total_paths):

    input_list = np.array(eigenvalues_by_path)
    obj = []
    
    i = 0
    for i in range(total_paths):
        for x in np.nditer(input_list[i], flags = ['external_loop'], order='F'):
            obj.append(x)

    del obj[::3]
    del obj[1::2]

    return obj

#Obtain the Fermi Level
def fermi_energy(filename3):
    with open(filename3) as fp:
        for i, line in enumerate(fp):
            if i == 5:
                line = line.strip()
                data = re.split(" +", line)
                E_Fermi = float(data[3])
            elif i > 5:
                break
    return E_Fermi

#Plots the Band Structure
def matplotlib_code(seek_path, total_paths, number_of_bands, sorted_EVALS, normalized_path, \
                    KPOINTS_BY_PATH, subdivisions_per_path, E_FERMI, new_path, path_magnitudes):
    style.use('default')
    sorted_EVALS = np.array(sorted_EVALS)
    ymin = np.min(sorted_EVALS - E_FERMI) - 1
    ymax = np.max(sorted_EVALS - E_FERMI) + 1
    #x_labels = new_path
    path_labels = seek_path
    
    fig,ax = plt.subplots(figsize=(10,5)) #fig size same as before
    gs = gridspec.GridSpec(1, len(normalized_path), wspace = 0, width_ratios = normalized_path)
    path_magnitudes.insert(0,0.0)
    
    #Stylize the Graph
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams['axes.xmargin'] = 0
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('font', size=10)          # controls default text sizes
    
    i = 0
    j = 0
    colors = ['k','k','k']
    
    for i in range(total_paths + 1):
        if i <= total_paths - 1:
            if i == 0:
                ax = plt.subplot(gs[i])   #xlabel = x_labels[i]
                ax.axes.xaxis.set_ticks([float(KPOINTS_BY_PATH[i,0])])
                ax.set_xticklabels([path_labels[i]])
                for j in range(number_of_bands):
                    eigenvalues = sorted_EVALS[i][j::number_of_bands]
                    plt.plot(KPOINTS_BY_PATH[i], eigenvalues - E_FERMI, c=colors[j % len(colors)], alpha=0.5)


                custom_ylim = (ymin, ymax)
                plt.setp(ax, ylim=custom_ylim)
                plt.ylabel('Energy (eV)')
                ax.yaxis.label.set_size(16)
                ax.axhline(y=0, label='Fermi Level', alpha=0.85, color = 'gray', ls='--')
                i += 1
            else:
                ax = plt.subplot(gs[i]) #xlabel = x_labels[i]
                ax.axes.yaxis.set_ticks([])
                ax.set_yticklabels([]) 
                ax.axes.xaxis.set_ticks([float(KPOINTS_BY_PATH[i,0])])    #([path_magnitudes[i]])
                ax.set_xticklabels([path_labels[i]])         #([path_labels[i]])
                for j in range(number_of_bands):
                    eigenvalues = sorted_EVALS[i][j::number_of_bands]
                    plt.plot(KPOINTS_BY_PATH[i], eigenvalues - E_FERMI, c=colors[j % len( colors)], alpha=0.5)

                custom_ylim = (ymin, ymax)
                plt.setp(ax, ylim=custom_ylim)
                ax.axhline(y=0, label='Fermi Level', alpha=0.85, color = 'gray', ls='--')
                i += 1
        else:
            ax = plt.subplot(gs[i], xlabel = 'DOS (1/eV)')
            ax.axes.yaxis.set_ticks([])
            ax.set_yticklabels([])
            ax.axes.xaxis.set_ticks([0.0])
            ax.set_xticklabels([path_labels[i]])
            
            ENERGY = read_doscar_data(DOSCAR_VASP_FILE)[0]
            DOS_U = read_doscar_data(DOSCAR_VASP_FILE)[1]
            #DOS_D = read_doscar_data(filename3)[2]
            E_Fermi = fermi_energy(DOSCAR_VASP_FILE)
            ENERGY = np.array(ENERGY)
            ENERGY_FERMI_SHIFTED = np.array(ENERGY - E_Fermi)
            DOS_U = np.array(DOS_U)
            
            ax.plot(DOS_U,ENERGY_FERMI_SHIFTED,color='k',alpha=0.70)
            custom_ylim = (ymin, ymax)
            plt.setp(ax, ylim=custom_ylim)
            ax.axhline(y=0, label='Fermi Level', alpha=0.85, color = 'gray', ls='--')
            
            #plt.plot(DOS_D,ENERGY_FERMI_SHIFTED,color=sns.color_palette()[0],label='Spin-Down')
            #plt.fill_between(ENERGY_FERMI_SHIFTED,np.zeros_like(DOS_U),DOS_U, where = (ENERGY_FERMI_SHIFTED < 0), color=sns.color_palette()[1],alpha=0.4)
            #plt.fill_between(ENERGY_FERMI_SHIFTED,DOS_D,np.zeros_like(DOS_D), where = (ENERGY_FERMI_SHIFTED < 0), color=sns.color_palette()[0],alpha=0.4)
    
    
    plt.subplots_adjust(wspace = 0.0)
    fig = plt.gcf()

    plt.rcParams.update({'font.family':'serif'})
    fig.suptitle(f'Band Structure and DOS for {SYSTEM_NAME}', fontsize=20)
    fig.supxlabel('High Symmetry Points', fontsize=16)

    if yes_or_no('Would you like to save the band structure plot to the current directory?'):
        plt.savefig(f"band_structure_and_dos_{SYSTEM_NAME}.pdf")

    return fig

def bs_plotting_tool(KPOINTS_VASP_FILE, EIGENVAL_VASP_FILE, DOSCAR_VASP_FILE):
    '''Obtain data from definitions in order to plot'''
    path_magnitudes = read_kpoints_file(KPOINTS_VASP_FILE)[0]
    new_path = read_kpoints_file(KPOINTS_VASP_FILE)[1]

    normalized_path = normalized_path_mag(KPOINTS_VASP_FILE)

    number_of_electrons = number_of_electrons_kpoints_bands(EIGENVAL_VASP_FILE)[0]
    number_of_kpoints = number_of_electrons_kpoints_bands(EIGENVAL_VASP_FILE)[1]
    number_of_bands = number_of_electrons_kpoints_bands(EIGENVAL_VASP_FILE)[2]

    subdivisions_per_path = subdivisions(KPOINTS_VASP_FILE)

    total_paths = number_of_paths(KPOINTS_VASP_FILE, EIGENVAL_VASP_FILE)

    KPOINTS_BY_PATH = obtain_KPOINTS_paths(EIGENVAL_VASP_FILE, subdivisions_per_path)

    eigenvalues_by_path = obtain_EVALS(EIGENVAL_VASP_FILE, subdivisions_per_path)

    sorted_EVALS = evals_sort(eigenvalues_by_path, total_paths)

    seek_path = []
    seek_path_string = input('Please enter the K-Points path used in the KPOINTS file, separated by dashes (e.g. L-G-X): ')
    new_string = seek_path_string.split('-')
    for i in range(total_paths+1):
        seek_path.append(new_string[i])
    '''Data obtained...Ready to Plot...'''
    E_FERMI = fermi_energy(DOSCAR_VASP_FILE)
    matplotlib_code(seek_path, total_paths, number_of_bands, sorted_EVALS, normalized_path, \
                KPOINTS_BY_PATH, subdivisions_per_path, E_FERMI, new_path, path_magnitudes)

    return number_of_electrons, number_of_bands, number_of_kpoints

def validate_system_name(name: str) -> str:
    if not re.match("^[a-zA-Z0-9_\-() ]+$", name):
        raise ValueError("Invalid system name. Only alphanumeric characters, spaces, dashes, parentheses and underscores are allowed.")
    return name

def get_validated_input(prompt: str, validation_function) -> str:
    while True:
        user_input = input(prompt)
        try:
            return validation_function(user_input)
        except ValueError as e:
            print(e)

def validate_float_within_range(value: str, min_value: float, max_value: float) -> float:
    try:
        float_value = float(value)
        if min_value <= float_value <= max_value:
            return float_value
        else:
            raise ValueError(f"Invalid input. Please enter a floating-point number within the range {min_value:.6f} to {max_value:.6f}.")
    except ValueError:
        raise ValueError("Invalid input. Please enter a floating-point number within the range specified.")

if __name__ == "__main__":

    SYSTEM_NAME = get_validated_input("What is the name of this system? ", validate_system_name)



#DOSCAR STUFF
    try:
        DOSCAR_VASP_FILE = 'DOSCAR'
    except FileNotFoundError:
        DOSCAR_VASP_FILE = input("Enter the path to your DOSCAR file: ")
    if yes_or_no('Would you like to run the DOS plotting tool? '):
        dos_plotting_tool(DOSCAR_VASP_FILE)
        interactive_dos_plotting_tool(DOSCAR_VASP_FILE, SYSTEM_NAME)
        
        

#BAND STRUCTURE STUFF
    try:
        KPOINTS_VASP_FILE = 'KPOINTS'
        EIGENVAL_VASP_FILE = 'EIGENVAL'
        DOSCAR_VASP_FILE = 'DOSCAR'
    except FileNotFoundError:
        KPOINTS_VASP_FILE = input("Enter the path to your KPOINTS file: ")
        EIGENVAL_VASP_FILE = input("Enter the path to your EIGENVAL file: ")
        DOSCAR_VASP_FILE = input("Enter the path to your DOSCAR file: ")
    
    #BAND STRUCTURE PLOTTING
    if yes_or_no('Would you like to run the band structure plotting tool? '):
        bs_plotting_tool(KPOINTS_VASP_FILE, EIGENVAL_VASP_FILE, DOSCAR_VASP_FILE)
