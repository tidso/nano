Here we will do the bandstructure calculation.

Start by creating a new directory called "band."

Copy the CHGCAR file from the "relax" directory (output) into this new directory.

Next, go back to the "relax" directory. Copy the contents of CONTCAR into the POSCAR file in the "band" directory.
(The CONTCAR file gives the atomic positions after geometric relaxation).

The POTCAR file can be copied directly from the "relax" folder as well.

Now, take a look at INCAR. You do not need to change anything. Note that ICHARG has been set to 11 and NSW is now 0. 

Open KPOINTS. You will not change anything here either. Note that it is written in reciprocal mode. 
This lets you set the path along which you would like to sample the kpoints. We will use the path: L-Γ-X.
You will want to copy or take note of the string "L-Γ-X" because we will use it in running the Python code.
(Alternatively, you can type L-G-X. Γ references the Gamma point located in reciprocal space. In the literature, Γ is more common to see than G).

Run the band jobscript.

Create a local folder on your computer. Name it something relevant like "Si Bandstructure."

COPY the DOSCAR, KPOINTS, and EIGENVAL files directly from PUTTY to this local folder. You should copy the actual files to this folder. Do not copy and paste the contents of each file into .txt files or some other format. (This is best accopmlished via WinSCP or FileZilla).

Once you have copied those output files over, open the IDSO_VASP_PLOTTING_TOOL in a Python IDE. I recommend Spyder via the Anaconda Distribution because that is where the script was written and tested. However, let me know what other IDE's work if you try something else.

Set the working directory of the Python IDE to be the local folder created previously (line 19).

Run the code. Name the system when prompted. "Silicon (Si) Diamond" would be a good name to use.

Run the plotting tool and save the plot by responding 'yes' to both prompts.

Next, run the band structure tool. You will need to tell the script what KPOINTS path was taken in the KPOINTS file located in the "band" folder. For this tutorial, we used L-Γ-X for our path through reciprocal space. Enter this path.

Save the band structure plot.

Notice that the colors for each band are printed. This is to help you identify cases of degeneracy. The bands are colored from bottom (blue) to top. If more colors are needed, it will cycle back and use blue again. However, 10 colors are provided, and many systems do not use more than 10 bands. 

All done :)
