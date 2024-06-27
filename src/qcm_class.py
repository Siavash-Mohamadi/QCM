import pandas as pd
import numpy as np
import ruptures as rpt
import scipy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter, butter, lfilter, filtfilt


class QCM:

    def __init__(self, data, smoothing=True, time_span=30, lowpass_freq= 0.001):
        self.data = data
        self.lpf = lowpass_freq
        self.smooth = smoothing
        self.time_span = time_span
        self.X = self.data.iloc[:, 0].apply(lambda x: x)
        self.X[0] = 1e-10 if self.X[0] == 0 else self.X[0]
        self.Y = self.data.iloc[:, 1]
        self.Y = pd.Series(self.filter(self.Y)) if smoothing else self.Y
        self.n_col = 2
        self.n_row = 1
        self.axs = None
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None
        self.model = None
        self.param = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.adj_r_squared = None
        self.y_model = None
        self.x_model = None
        self.func_list = None
        self.model_name = None
        self.index = None
        self.time_start = None
        self.upper_bound = None
        self.lower_bound = None
        self.parameters = None
        self.residuals = None
        self.cut_point_indices = None
        self.bounds = None

        self.func_list = pd.DataFrame({
            'FunctionName': ['Avrami', 'Pseudo-First-Order', 'Lagged_Pseudo-First-Order', 'Elovich', 
                             'Exponential Growth', 'Boltzmann Sigmoidal (Free Start)', 'Boltzmann Sigmoidal (Fixed Start)', 
                             'Boltzmann Sigmoidal (B=1, Free Start)', 'Boltzmann Sigmoidal (B=1, Fixed Start)', 
                             'Double Exponential', 'Double Exponential (Non-symmetric)', 'Pseudo-Second-Order'],
            'FunctionObject': [self.avrami, self.pseudo_first_order, self.lagged_pseudo_first_order, self.elovich,
                               self.exponential_growth, self.boltzmann_sigmoidal_free_start, self.boltzmann_sigmoidal_fixed_start,
                               self.boltzmann_sigmoidal_symmetric_free_start, self.boltzmann_sigmoidal_symmetric_fixed_start,
                               self.double_exponential, self.non_symmetric_double_exponential, self.pseudo_second_order]
        })

    def avrami(self, x, k_av, n_av):
        self.parameters = ['k_av', 'n_av']
        return (self.y_max - self.y_min) * (1 - np.exp(-k_av * (x ** n_av)))

    def pseudo_first_order(self, x, k_p):
        self.parameters = ['k_p']
        return self.y_max * (1 - np.exp(-k_p * x))

    def lagged_pseudo_first_order(self, t, k_p, t_i):
        self.parameters = ['k_p', 't_i']
        return np.maximum(0, self.y_max * (1 - np.exp(-k_p * (t - t_i))))

    def pseudo_second_order(self, x, k_2):
        self.parameters = ['k_2']
        return (self.y_max**2) * ((k_2 * x)/(1 + self.y_max * k_2 * x))

    def elovich(self, x, alpha, beta):
        self.parameters = ['alpha', 'beta']
        return (np.log(x) / beta) + (np.log(alpha * beta) / beta)

    def exponential_growth(self, x, a_0, k_eg):
        self.parameters = ['a_0', 'k_eg']
        return a_0 * np.exp(k_eg * x)

    def boltzmann_sigmoidal_free_start(self, x, a_b1, a_b2, b, ti, r_bs):
        self.parameters = ['a_b1', 'a_b2', 'b', 't_i', 'r_bs']
        return a_b1 + ((a_b2 - a_b1) / (b + np.exp((ti - x) / r_bs)))

    def boltzmann_sigmoidal_fixed_start(self, x, a_b2, b, ti, r_bs):
        self.parameters = ['a_b2', 'b', 't_i', 'r_bs']
        return self.y_min + ((a_b2 - self.y_min) / (b + np.exp((ti - x) / r_bs)))

    def boltzmann_sigmoidal_symmetric_free_start(self, x, a_b1, a_b2, ti, r_bs):
        self.parameters = ['a_b1', 'a_b2', 't_i', 'r_bs']
        return a_b1 + ((self.y_max - a_b1) / (1 + np.exp((ti - x) / r_bs)))

    def boltzmann_sigmoidal_symmetric_fixed_start(self, x, a_b2, ti, r_bs):
        self.parameters = ['a_b2', 't_i', 'r_bs']
        return self.y_min + ((a_b2 - self.y_min) / (1 + np.exp((ti - x) / r_bs)))

    def double_exponential(self, t, a_1, b_1, k_1, a_2, b_2, k_2):
        self.parameters = ['a_1', 'b_1', 'k_1', 'a_2', 'b_2', 'k_2']
        return a_1 * (1 - b_1 * np.exp(-k_1 * t)) + a_2 * (1 - b_2 * np.exp(-k_2 * t))

    def non_symmetric_double_exponential(self, t, a_1, b_1, k_1, a_2, b_2, k_2):
        self.parameters = ['a_1', 'b_1', 'k_1', 'a_2', 'b_2', 'k_2']

        return a_1 * (1 - b_1 * np.exp(-k_1 * t)) + a_2 * (1 - b_2 * np.exp(k_2 * t))

    def pre_cut(self, start, end):
        self.data = self.data[(self.data.iloc[:, 0] > start) & (self.data.iloc[:, 0] < end)].reset_index(
            drop=True)
        self.time_start = start
        self.time_span = end


    def baseline_correction(self, baseline_start, baseline_end):
        df_correction = self.data[(self.data.iloc[:, 0] > baseline_start) & (self.data.iloc[:, 0] < baseline_end)].reset_index(drop=True)
        slope, intercept, _, _, _ = scipy.stats.linregress(df_correction.iloc[:, 0],
                                                           df_correction.iloc[:, 1])
        self.data.iloc[:, 1] -= slope * self.data.iloc[:, 0] + intercept
        self.Y -= slope * self.X + intercept


    def plot_raw(self):
        filtered_data = self.data[self.data.iloc[:, 0] < self.time_span]
        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(filtered_data.iloc[:, 0], filtered_data.iloc[:, 1], zorder=2, linewidth=0.2)
        plt.title('Raw Data')
        plt.xlabel('Time (min)')
        plt.ylabel('Mass')
        plt.grid(zorder=1)
        plt.tight_layout()
        plt.show()

    def index_to_row_col(self, index):
        row = index // self.n_col
        col = index % self.n_col
        return row, col

    def qcm_plot(self, i, xx, yy, residuals):
        row, col = self.index_to_row_col(i)

        # Create a new subplot with specified axes limits
        self.axs[row, col] = plt.subplot(self.n_row, self.n_col, i + 1)

        # Plot data on the first subplot (top)
        self.axs[row, col].scatter(xx, yy, label='Data', color='blue')
        self.axs[row, col].plot(self.x_model, self.y_model, color='r', label=str(self.model_name))

        # Set limits for x and y axes
        self.axs[row, col].set_xlim(min(xx), max(xx))
        self.axs[row, col].set_ylim(min(yy), max(yy))

        # Set labels for the x and y axes
        self.axs[row, col].set_xlabel('Time (min)')
        self.axs[row, col].set_ylabel('Mass (ng/$cm^2$)', color='blue')

        # Set Title
        self.axs[row, col].set_title(f'Time = 0 - {round(self.x_max)} min', fontweight="bold")

        # Create a secondary y-axis on the right
        ax2 = self.axs[row, col].twinx()

        # Plot the secondary function on the secondary y-axis
        ax2.plot(xx, residuals, color='g', label='Residuals')

        # Set labels for the secondary y-axis
        ax2.set_ylabel('Residuals', color='green')

        # Merge the legends from the primary and secondary y-axes into a single legend on the primary axis
        lines, labels = self.axs[row, col].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.axs[row, col].legend(lines + lines2, labels + labels2, loc='lower right')

    def filter(self, Y):
        lowpass_frequency = self.lpf  # Hz
        b_low, a_low = butter(N=2, Wn=lowpass_frequency, btype="lowpass", fs=0.67, output='ba')
        Y_filtered = filtfilt(b_low, a_low, Y)
        return Y_filtered - Y_filtered[0]


    def change_point_detection(self):
        df_baseline = pd.DataFrame()
        df_baseline = pd.concat([self.X, self.Y], axis=1)
        df_baseline.columns = ['time', 'mass']
        mask = (df_baseline['time'] < self.time_span) & (df_baseline['time'] > self.time_start)
        df_baseline = df_baseline[mask]
        self.X = df_baseline['time']
        self.Y = df_baseline['mass']
        # First derivative (working with filtered data)
        df_baseline['Mass_Derivative'] = np.gradient(df_baseline['mass'], df_baseline['time'])
        window_length = min(400, len(df_baseline['Mass_Derivative']) - 1)
        df_baseline['Mass_Derivative'] = savgol_filter(df_baseline['Mass_Derivative'], window_length, 2)
        df_baseline['2nd_Mass_Derivative'] = np.gradient(df_baseline['Mass_Derivative'], df_baseline['time'])
        df_baseline['2nd_Mass_Derivative'] = savgol_filter(df_baseline['2nd_Mass_Derivative'], window_length, 2)

        plt.style.use('default')

        # Create a figure and the first y-axis (left)
        fig, ax1 = plt.subplots(figsize=(25, 10))

        # Plot the Mass Derivative on the left axis
        ax1.plot(df_baseline['time'], df_baseline['Mass_Derivative'], label='1st Derivative', color='blue')
        ax1.set_xlabel('Time')

        ax1.set_ylabel('1st Derivative', color='blue', fontsize=15)
        ax1.grid(True)

        # Create a second y-axis (right)
        ax2 = ax1.twinx()

        # Plot the 2nd Mass Derivative on the right axis
        ax2.plot(df_baseline['time'], df_baseline['2nd_Mass_Derivative'], label='2nd Derivative', color='green')
        ax2.set_ylabel('2nd Derivative', color='green', fontsize=15)
        ax2.grid(True, color='green', linestyle='--')

        # Title and legend
        plt.title('Time Derivatives of Mass', fontsize=25)
        plt.xticks(np.arange(min(df_baseline['time']), max(df_baseline['time']) + 1, 5.0))

        # Combine the legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=15)

        # Use change point detection to find the start of the rise
        self.cut_point_indices = self.find_change_points(df_baseline['Mass_Derivative'])

        # Plot the detected change points
        ax1.plot(df_baseline['time'].iloc[self.cut_point_indices], df_baseline['Mass_Derivative'].iloc[self.cut_point_indices],
                 'ro', label='Change Points')

        # Show the plot
        plt.show()


    def baseline_cut(self):
        self.X = self.X[self.cut_point_indices[0]:self.cut_point_indices[1]].reset_index(drop=True)
        self.X = self.X - self.X[0]
        self.Y = self.Y[self.cut_point_indices[0]:self.cut_point_indices[1]].reset_index(drop=True)
        self.Y = self.Y - self.Y[0]

    def find_change_points(self, data):
        # Using BottomUp method for change point detection
        model = rpt.BottomUp(model="rbf").fit(data.values)
        change_points = model.predict(n_bkps=2, pen=3)

        # Ensure the change points are within the valid range
        valid_change_points = [point for point in change_points if point < len(data)]
        return valid_change_points



    def fit_and_plot(self, model_name, save=False, confidence_interval=False):
        self.model_name = model_name
        self.index, _ = np.where(self.func_list == self.model_name)
        if len(self.index) == 0:
            raise ValueError(f"Model '{self.model_name}' not found in list")
        self.model = self.func_list.loc[self.index[0], 'FunctionObject']
        x = self.X[self.X < self.time_span]
        y = self.Y[self.X < self.time_span]
        self.param = [0, 0, 0, 0]
        self.y_max = np.max(y)
        self.y_min = np.min(y)
        self.x_max = np.max(x)
        self.x_min = np.min(x)

        # Adding constraint to the ti parameter for Boltzman Sigmoidal model
        if self.model_name == 'Boltzmann Sigmoidal (Free Start)':
            self.bounds = ((-np.inf, self.y_max * (1 - 0.5), -np.inf, self.x_min, -np.inf),
                           (np.inf, self.y_max * (1 + 0.5), np.inf, self.x_max, np.inf))
            popt, pcov = curve_fit(self.model, x, y, full_output=False, bounds=self.bounds)

        elif self.model_name == 'Boltzmann Sigmoidal (Fixed Start)':
            self.bounds = ((self.y_max * (1 - 0.2), -np.inf, self.x_min, -np.inf),
                           (self.y_max * (1 + 0.2), np.inf, self.x_max, np.inf))
            popt, pcov = curve_fit(self.model, x, y, full_output=False, bounds=self.bounds)

        elif self.model_name == 'Double Exponential (Non-symmetric)':
            self.bounds = ((0, -np.inf, 0, -np.inf, -np.inf, 0),
                           (np.inf, np.inf, np.inf, 0, np.inf, np.inf))
            popt, pcov = curve_fit(self.model, x, y, full_output=False, bounds=self.bounds)

        else:
            popt, pcov = curve_fit(self.model, x, y, full_output=False)

        # Confidence Itervals

        # Calculate standard errors from the diagonal elements of the covariance matrix
        perr = np.sqrt(np.diag(pcov))

        # Construct confidence intervals (assuming a 95% confidence level)
        alpha = 0.05
        z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
        conf_intervals = z_critical * np.sqrt(np.diag(pcov))

        # Generate x values for plotting
        self.x_model = np.linspace(np.min(x), np.max(x), 100)
        self.y_model = self.model(x, *popt)
         #Use the confidence intervals to plot shaded regions around the fitted curve
        self.upper_bound = self.model(x, *(popt + conf_intervals))
        self.lower_bound = self.model(x, *(popt - conf_intervals))


        self.param = popt
        self.residuals = y - self.model(x, *popt)
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        self.r_squared = r_squared

        # Calculate AIC and BIC
        n = len(y)
        self.aic = n * np.log(np.sum(self.residuals**2) / n) + 2 * len(self.param)
        self.bic = n * np.log(np.sum(self.residuals**2) / n) + len(self.param) * np.log(n)

        # Calculate Reduced Chi-square
        self.adj_r_squared = 1 - (((1-self.r_squared)*(n-1))/(n-len(popt)-1))

        plt.style.use('default')

        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
        ax2 = ax1.twinx()
        ax2.grid(True)
        # Plot data on the first subplot (top)
        ax1.scatter(x, y, label='Data', color='blue', s=2)
        ax1.plot(x, self.y_model, color='r', label=str(self.model_name))
        if confidence_interval:
            ax1.fill_between(x, self.lower_bound, self.upper_bound, color='black', alpha=0.5, label='95% Confidence Interval')

        # padding_percent = 10  # Adjust as needed
        # y_range = np.max(y) - np.min(y)
        # padding = y_range * padding_percent / 100
        # # Set y-axis limits with padding
        # ax1.set_ylim([np.min(y) - padding, np.max(y) + padding])

        # Set labels for the x and y axes
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Mass (ng/$cm^2$)', color='blue')

        # Plot the secondary function on the secondary y-axis
        ax2.plot(x[:], 100 * self.residuals[:] / y[:], color='g', label='Residuals %')
        ax2.set_ylim([-100, 100])

        # Set labels for the secondary y-axis
        ax2.set_ylabel('Residuals %', color='green')

        # Solve the issue of ax2 grid overlapping the ax1 graph
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # Merge the legends from the primary and secondary y-axes into a single legend on the primary axis
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()


        self.r_squared = pd.DataFrame([self.r_squared], columns=['R-Squared'])
        self.y_max = pd.DataFrame([self.y_max], columns=['Max'])
        self.y_min = pd.DataFrame([self.y_min], columns=['Min'])
        self.aic = pd.DataFrame([self.aic], columns=['AIC'])
        self.bic = pd.DataFrame([self.bic], columns=['BIC'])
        self.adj_r_squared = pd.DataFrame([self.adj_r_squared], columns=['Adjusted R-Squared'])

        self.param = pd.DataFrame([self.param], columns=self.parameters)
        self.param = pd.concat([self.param, self.y_min, self.y_max, self.r_squared, self.aic, self.bic, self.adj_r_squared], axis=1)
        print(self.param)

        self.param.to_csv('parameters.csv', index=False)

        x_df = pd.DataFrame(x)
        y_df = pd.DataFrame(y)
        residuals_df = pd.DataFrame(self.residuals)
        y_model_df = pd.DataFrame(self.y_model)
        # Concatenate DataFrames along columns axis
        result_df = pd.concat([x_df, y_df, residuals_df, y_model_df], axis=1)
        result_df.columns = ['Time', 'Mass', 'Residials', 'Model']
        result_df.to_csv('output.csv', index=False)


        if save:
            plt.savefig(f'Fitted by {self.model_name}.png', dpi = 600)


