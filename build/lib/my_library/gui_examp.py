import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
import RSPAnalyzer as rs



class SupernovaAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RSPAnalyzer")
        self.create_widgets()

    def create_widgets(self):
        # Existing widgets
        tk.Label(self.root, text="CSV File Path:").grid(row=0, column=0, padx=10, pady=10)
        self.csv_path_entry = tk.Entry(self.root, width=50)
        self.csv_path_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Distance to SN (Mpc):").grid(row=1, column=0, padx=10, pady=10)
        self.dist_entry = tk.Entry(self.root)
        self.dist_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Initial k:").grid(row=2, column=0, padx=10, pady=10)
        self.init_k_entry = tk.Entry(self.root)
        self.init_k_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Initial m:").grid(row=3, column=0, padx=10, pady=10)
        self.init_m_entry = tk.Entry(self.root)
        self.init_m_entry.grid(row=3, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Initial gamma:").grid(row=4, column=0, padx=10, pady=10)
        self.init_gamma_entry = tk.Entry(self.root)
        self.init_gamma_entry.grid(row=4, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Number of Walkers:").grid(row=5, column=0, padx=10, pady=10)
        self.nwalkers_entry = tk.Entry(self.root)
        self.nwalkers_entry.grid(row=5, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Number of Steps (chain size, at least 10000):").grid(row=6, column=0, padx=10, pady=10)
        self.nsteps_entry = tk.Entry(self.root)
        self.nsteps_entry.grid(row=6, column=1, padx=10, pady=10)

        # New checkbutton
        tk.Label(self.root, text="Fit m:").grid(row=7, column=0, padx=10, pady=10)
        self.fit_m_var = tk.BooleanVar()
        tk.Checkbutton(self.root, variable=self.fit_m_var).grid(row=7, column=1, padx=10, pady=10)

        # Shift existing widgets down
        tk.Label(self.root, text="Show Spectra:").grid(row=8, column=0, padx=10, pady=10)
        self.spectra_var = tk.BooleanVar()
        tk.Checkbutton(self.root, variable=self.spectra_var).grid(row=8, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Show Light Curve:").grid(row=9, column=0, padx=10, pady=10)
        self.light_curve_var = tk.BooleanVar()
        tk.Checkbutton(self.root, variable=self.light_curve_var).grid(row=9, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Legend for Spectra:").grid(row=10, column=0, padx=10, pady=10)
        self.slegend_var = tk.BooleanVar()
        tk.Checkbutton(self.root, variable=self.slegend_var).grid(row=10, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Legend for Light Curve:").grid(row=11, column=0, padx=10, pady=10)
        self.lclegend_var = tk.BooleanVar()
        tk.Checkbutton(self.root, variable=self.lclegend_var).grid(row=11, column=1, padx=10, pady=10)

        tk.Label(self.root, text="X Lim (min,max):").grid(row=12, column=0, padx=10, pady=10)
        self.xlim_entry = tk.Entry(self.root)
        self.xlim_entry.grid(row=12, column=1, padx=10, pady=10)

        tk.Label(self.root, text="Y Lim (min,max):").grid(row=13, column=0, padx=10, pady=10)
        self.ylim_entry = tk.Entry(self.root)
        self.ylim_entry.grid(row=13, column=1, padx=10, pady=10)

        # Run button
        tk.Button(self.root, text="Run", command=self.run_analysis).grid(row=14, column=0, columnspan=3, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_path_entry.delete(0, tk.END)
            self.csv_path_entry.insert(0, file_path)

    def run_analysis(self):
        try:
            csv_path = self.csv_path_entry.get()
            dist_to_sn = float(self.dist_entry.get())
            init_k = float(self.init_k_entry.get())
            init_m = float(self.init_m_entry.get())
            init_gamma = float(self.init_gamma_entry.get())
            nwalkers = int(self.nwalkers_entry.get())
            nsteps = int(self.nsteps_entry.get())
            fit_m = self.fit_m_var.get()
            spectra = self.spectra_var.get()
            light_curve = self.light_curve_var.get()
            slegend = self.slegend_var.get()
            lclegend = self.lclegend_var.get()

            xlim = self.parse_limit(self.xlim_entry.get())
            ylim = self.parse_limit(self.ylim_entry.get())

            # Run the main function
            sampler, mcmc_params, flat_samples = rs.run_prog(
                csv_path=csv_path,
                dist_to_sn=dist_to_sn*rs.MPC,
                init_k=init_k,
                init_m=init_m,
                init_gamma=init_gamma,
                nwalkers=nwalkers,
                nsteps=nsteps,
                fit_m = fit_m,
                spectra=spectra,
                light_curve=light_curve,
                slegend=slegend,
                lclegend=lclegend,
                xlim=xlim,
                ylim=ylim
            )

            plt.show()  # Show the plots

            # Optionally, display a success message
            messagebox.showinfo("Success", "Analysis complete!")
        except Exception as e:
            messagebox.showerror("Error:" , str(e))

    def parse_limit(self, limit_str):
        if limit_str:
            try:
                limits = tuple(map(float, limit_str.split(',')))
                if len(limits) == 2:
                    return limits
            except ValueError:
                pass
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = SupernovaAnalysisGUI(root)
    root.mainloop()
