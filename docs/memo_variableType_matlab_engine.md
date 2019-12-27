# Variable types of Python for using matlab.engine 

- Author: Takanori Sugiyama
- Date: December 25, 2019
- Purpose of this document:\
To explain the variable types in Python scripts for calling MATLAB functions using matlab.engine
- References
  https://jp.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html

  https://jp.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html


1. $d$, the dimention of the quantum system of interest.
   - Type: integer (uint8)  

2. num_state, $N_{\mathrm{s}}$, the number of states used.
   - Type: integer (uint16)

3. list_state_vec, $\{ |\rho_j \rangle\!\rangle \}_{j=0}^{N_{\mathrm{s}}-1}$, the list of vectorized density matrices
   - Type: list of list of complex double (complex float in Python)
   - Math:$|\rho\rangle\!\rangle \in \mathbf{C}^{d^2}$ is the vectorization of a matrix $\rho \in \mathbf{C}^{d \times d}$ with respect to the computational basis (row-major-order). 
   - Size: $N_{\mathrm{s}} \times d^2$

4. num_povm, $N_{\mathrm{p}}$, the number of POVMs used.
    - Type: integer (unit16)

5. num_outcome, $M$, the number of possible outcomes for each POVM. Assumed to be common among all POVMs.
    - Type: integer (unit 8)

6. list_povm_vec, $\{ | \mathbf{\Pi}_{j} \rangle\!\rangle \}_{j=0}^{N_{\mathrm{p}}-1}$, the list of vectorized POVMS
    - Type: list of list of list of complex double (complex float in Python)
    - Math:$|\mathbf{P}_j\rangle\!\rangle = \{ | \Pi_{j, \omega}\rangle\!\rangle \}_{\omega = 0}^{M -1}$, $| \Pi_{j, \omega} \rangle\!\rangle \in \mathbf{C}^{d^2}$ is the vectorization of a POVM element $\Pi_{\omega} \in \mathbf{C}^{d \times d}$ with respect to the computational basis (row-major-order). 
    - Size: $N_{\mathrm{p}} \times M \times d^2$

7. num_schedule, $N_{\mathrm{schedule}}$, the number of schedules
    - Type: integer (unit32)

8. list_schedule
    - Type: list of list of integer (unit16)
    - Math: $j = (j_{\mathrm{s}}, j_{\mathrm{p}})$ is a pair of integer (unit16)
    - Size: $N_{\mathrm{schedule}} \times 2$

8. list_weight
    - Type: list of list of list of real double (real float in Python)
    - Math: $\{ W_{j} \}_{j=0}^{N_{\mathrm{schedule}-1}}$. $W_j \in \mathbf{R}^{M \times M}$.
    - Size: $N_{\mathrm{schedule}} \times M \times M$

9. list_num_data, $\{ N_{j} \}_{j=0}^{N_{\mathrm{schedule}}-1}$
    - Type: list of integer (uint32)
    - Size: $N_{\mathrm{schedule}}$
    
10. list_probdist, $\{ \bm{p}_{j} \}_{j=0}^{N_{\mathrm{schedule}-1}}$
    - Type: list of list of real double (real float in Python)
    - Math: $\bm{p}_{j} = \{ p(\omega | j) \}_{\omega = 0}^{M-1}$ is a probability distribution.
    - Size: $\N_{\mathrm{schedule}} \times M$

11. $k$, the length of the gate sequence
    - Type: integer (uint16)

12. $L_0$, the target Lindblad generator (accumulated)
    - Type: list of list of complex double (complex float in Python)
    - Math: $L_0 \in \mathbf{C}^{d^2 \times d^2}$
    - Size: $d^2 \times d^2$

13. $\bf{B}=\{ B_{\alpha} \}_{\alpha=0}^{d^2 -1}$, the matrix basis for the vectorization and the Hilbert-Schmidt representation.
    - Type: list of list of list of complex double (complex float in Python)
    - Math: $B_{\alpha} \in \mathbf{C}^{d \times d}$,
            Assumed to be Hermitian, orthonormalied, and the 0th element proportionals to the identity matrix, i.e., $B_{\alpha} = B_{\alpha}^{\dagger}$, $\mathrm{Tr}[B_{\alpha}^{\dagger} B_{\beta}] = \delta_{\alpha\beta}$, and $B_{0} = \mathbf{1}/\sqrt{d}$.

