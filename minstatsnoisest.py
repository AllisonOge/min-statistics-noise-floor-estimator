# @author Allison
import numpy as np


class NoiseEstimator:
    """A minimum statistics noise estimator,
    after instantiation of this class, pass the normalized fft samples, it should be noted that the
    initial noise power should be passed at instantiation"""

    def __init__(self, size, initial_noise_power):
        """initialize constructor"""
        self.__noise_est = 20 * np.log10(initial_noise_power)
        self.__last_psd_est = 0
        self.__prev_psd_est = np.ones(size) * initial_noise_power
        self.__prev_noise_est = np.zeros(size)
        self.__fmoment = 0
        self.__smoment = 0
        self.__eq_deg_per_frame = np.zeros(size)
        self.__last_alpha_cor = 0
        self.__alpha_arr = np.zeros(size)

    def compute(self, lin_fft):
        """Compute the noise estimate for each subsamples frame of FFT samples"""
        avg_norm = sum(self.__eq_deg_per_frame) / len(self.__eq_deg_per_frame)
        print("The average normalized variance in PSD samples is ", avg_norm)
        alpha_max = 0.96  # upper limit on alpha_var
        # compute short term psd for each frame
        x = sum(self.__prev_psd_est) / sum(abs(lin_fft) ** 2) - 1
        #         print("x is: {}".format(x))
        corfac_term = 0.3 * max(1 / (1 + x ** 2), 0.7) # compute correction factor for alpha
        b_corfac = 1 + 2.12 * np.sqrt(avg_norm)
        # in each frame
        for n in range(len(lin_fft)):
            # compute optimum psd estimate factor with previous estimated noise sample
            alpha_opt = 1 / (1 + (self.__last_psd_est / 10 ** (self.__noise_est / 10) - 1) ** 2)
            self.__last_alpha_cor = 0.7 * self.__last_alpha_cor + corfac_term
            alpha_var = alpha_max * self.__last_alpha_cor * alpha_opt
            beta_var = min(alpha_var ** 2, 0.8)
            self.__last_psd_est = alpha_var * self.__last_psd_est + (1 - alpha_var) * abs(lin_fft[n]) ** 2
            # compute first moment of psd estimate
            self.__fmoment = beta_var * self.__fmoment + (1 - beta_var) * self.__last_psd_est
            # compute second moment of psd estimate
            self.__smoment = beta_var * self.__smoment + (1 - beta_var) * self.__last_psd_est ** 2
            # estimated variance
            est_var = abs(self.__smoment - self.__fmoment)
            # equivalent degree of freedom
            eq_deg = 10 * np.log10(0.5 * est_var) - self.__noise_est
            eq_deg = min(10 ** (eq_deg / 10), 0.5)  # should be greater than or equal to two
            # find minimum sample in 3 of 8 samples
            win_min = 7 * len(lin_fft) // 8
            # computing inverse bias
            eq_deg_tilda = (1 / eq_deg - 2 * 0.91) / (1 - 0.91)
            bias = (win_min - 1) * 2 / abs(eq_deg_tilda)
            bias = 1 + bias
            # append psd estimated samples for a frame
            if n == 0:  # for the first sample in a frame
                noise_range = self.__prev_noise_est[len(lin_fft) - win_min - 1:]  # get win_len - 1 previous samples

            noise_range = np.append(noise_range, 10 * np.log10(b_corfac * bias * self.__last_psd_est))
            # db constraints between 0.8-5dB
            if avg_norm < 0.03:
                noise_slope_max = 9.03
            elif avg_norm < 0.05:
                noise_slope_max = 6.02
            elif avg_norm < 0.06:
                noise_slope_max = 3.01
            else:
                noise_slope_max = 0.8
            # slide through 3 of 8 samples
            min_stats = min(noise_range[n:n + win_min])
            self.__noise_est = min_stats + noise_slope_max
            # reassign and append new values
            self.__prev_psd_est = np.append(self.__prev_psd_est[1:], self.__last_psd_est)
            self.__prev_noise_est = np.append(self.__prev_noise_est[1:], self.__noise_est)
            self.__eq_deg_per_frame = np.append(self.__eq_deg_per_frame[1:], eq_deg)
            self.__alpha_arr = np.append(self.__alpha_arr[1:], alpha_var)
        # return computed values for the set of PSD estimates received
        return self.__prev_noise_est

    def getalpha(self):
        """Get the smoothening factor of the  noise estimator, this is used for observation purposes"""
        return self.__alpha_arr
