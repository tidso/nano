Here we will do the bandstructure calculation.

Start by copying the CHGCAR file from the "relax" folder (output) into this folder.

Next, go back to the "relax" folder. Copy the contents of CONTCAR into the POSCAR file in the "band" folder.
(The CONTCAR file gives the atomic positions after geometric relaxation).

The POTCAR file can be copied directly from the "relax" folder as well.

Now, take a look at INCAR. You do not need to change anything. Note that ICHARG has been set to 11 and NSW is now 0. 

Open KPOINTS. You will not change anything here either. Note that it is written in reciprocal mode. 
This lets you set the path along which you would like to sample the kpoints. We will use the path: L-Γ-X.
You will want to copy or take note of the string "L-Γ-X" because we will use it in running the Python code.
(Alternatively, you can type L-G-X. Γ references the Gamma point located in reciprocal space. In the literature, Γ is more common to see than G).
