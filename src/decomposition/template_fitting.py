"""Template Fitting Decomposition for PLR Waveforms.

Fits physiologically-constrained basis functions to PLR data:
- Transient/Phasic: Fast M-pathway response
- Sustained/Tonic: Maintained P-pathway response
- PIPR: Post-illumination pupil response (melanopsin)

Reference: Kelbsch et al. 2019, Kawasaki et al. 2002
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TemplateResult:
    """Result from template fitting decomposition."""

    phasic: NDArray[np.float64]  # Phasic component waveform
    sustained: NDArray[np.float64]  # Sustained component waveform
    pipr: NDArray[np.float64]  # PIPR component waveform
    baseline: float  # Baseline offset
    amplitudes: dict[str, float]  # Component amplitudes
    rmse: float  # Reconstruction RMSE
    fitted: NDArray[np.float64]  # Reconstructed waveform


class TemplateFitting:
    """Template-based decomposition of PLR waveforms.

    Fits physiological basis functions (transient, sustained, PIPR) to
    the observed pupil waveform using least squares.

    Parameters
    ----------
    blue_onset : float
        Time of blue stimulus onset (seconds)
    blue_offset : float
        Time of blue stimulus offset (seconds)
    red_onset : float
        Time of red stimulus onset (seconds)
    red_offset : float
        Time of red stimulus offset (seconds)
    """

    def __init__(
        self,
        blue_onset: float = 15.5,
        blue_offset: float = 24.5,
        red_onset: float = 46.5,
        red_offset: float = 55.5,
    ):
        self.blue_onset = blue_onset
        self.blue_offset = blue_offset
        self.red_onset = red_onset
        self.red_offset = red_offset

        # Time constants for templates
        self.tau_phasic_rise = 0.3
        self.tau_phasic_decay = 0.5
        self.tau_sustained_rise = 2.0
        self.tau_sustained_decay = 2.0
        self.tau_pipr = 15.0

    def _create_single_template(
        self,
        time_vector: NDArray[np.float64],
        onset: float,
        offset: float,
        tau_rise: float,
        tau_decay: float,
    ) -> NDArray[np.float64]:
        """Create a single stimulus-response template.

        Models pupil constriction as exponential approach during stimulus
        followed by exponential recovery after offset.
        """
        n_t = len(time_vector)
        template = np.zeros(n_t)
        t_from_onset = time_vector - onset
        t_from_offset = time_vector - offset

        # During stimulus: exponential approach to plateau
        stim_mask = (t_from_onset >= 0) & (t_from_offset <= 0)
        template[stim_mask] = 1 - np.exp(-t_from_onset[stim_mask] / tau_rise)

        # After stimulus: exponential decay from plateau
        post_mask = t_from_offset > 0
        plateau = 1 - np.exp(-(offset - onset) / tau_rise)
        template[post_mask] = plateau * np.exp(-t_from_offset[post_mask] / tau_decay)

        return -template  # Negative = constriction

    def _create_pipr_template(
        self, time_vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Create PIPR (post-illumination pupil response) template.

        Models the slow melanopsin-driven sustained constriction after
        light offset.
        """
        pipr = np.zeros(len(time_vector))

        # PIPR after blue stimulus
        t_blue = time_vector - self.blue_offset
        blue_mask = t_blue > 0
        pipr[blue_mask] -= np.exp(-t_blue[blue_mask] / self.tau_pipr)

        # PIPR after red stimulus
        t_red = time_vector - self.red_offset
        red_mask = t_red > 0
        pipr[red_mask] -= np.exp(-t_red[red_mask] / self.tau_pipr)

        return pipr

    def create_templates(
        self, time_vector: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Create normalized template waveforms.

        Parameters
        ----------
        time_vector : ndarray
            Time points in seconds

        Returns
        -------
        phasic, sustained, pipr : tuple of ndarray
            Normalized template waveforms
        """
        # Phasic template (fast response to both stimuli)
        phasic = self._create_single_template(
            time_vector,
            self.blue_onset,
            self.blue_offset,
            self.tau_phasic_rise,
            self.tau_phasic_decay,
        ) + self._create_single_template(
            time_vector,
            self.red_onset,
            self.red_offset,
            self.tau_phasic_rise,
            self.tau_phasic_decay,
        )

        # Sustained template (slower, maintained response)
        sustained = self._create_single_template(
            time_vector,
            self.blue_onset,
            self.blue_offset,
            self.tau_sustained_rise,
            self.tau_sustained_decay,
        ) + self._create_single_template(
            time_vector,
            self.red_onset,
            self.red_offset,
            self.tau_sustained_rise,
            self.tau_sustained_decay,
        )

        # PIPR template
        pipr = self._create_pipr_template(time_vector)

        # Normalize templates
        phasic = phasic / (np.abs(phasic).max() + 1e-10)
        sustained = sustained / (np.abs(sustained).max() + 1e-10)
        pipr = pipr / (np.abs(pipr).max() + 1e-10)

        return phasic, sustained, pipr

    def fit(
        self, waveform: NDArray[np.float64], time_vector: NDArray[np.float64]
    ) -> TemplateResult:
        """Fit templates to a single waveform.

        Parameters
        ----------
        waveform : ndarray
            PLR waveform (pupil size as % of baseline or mm)
        time_vector : ndarray
            Time points in seconds

        Returns
        -------
        TemplateResult
            Decomposition results with component waveforms and amplitudes
        """
        # Create templates
        phasic_template, sustained_template, pipr_template = self.create_templates(
            time_vector
        )

        # Design matrix: [phasic, sustained, pipr, constant]
        X = np.column_stack(
            [
                phasic_template,
                sustained_template,
                pipr_template,
                np.ones(len(time_vector)),
            ]
        )

        # Fit via least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(X, waveform, rcond=None)

        # Extract results
        phasic_amp, sustained_amp, pipr_amp, baseline = coeffs

        # Scale templates by amplitudes
        phasic_component = phasic_template * phasic_amp
        sustained_component = sustained_template * sustained_amp
        pipr_component = pipr_template * pipr_amp

        # Fitted waveform
        fitted = X @ coeffs

        # RMSE
        rmse = np.sqrt(np.mean((waveform - fitted) ** 2))

        return TemplateResult(
            phasic=phasic_component,
            sustained=sustained_component,
            pipr=pipr_component,
            baseline=baseline,
            amplitudes={
                "phasic": phasic_amp,
                "sustained": sustained_amp,
                "pipr": pipr_amp,
            },
            rmse=rmse,
            fitted=fitted,
        )

    def fit_batch(
        self, waveforms: NDArray[np.float64], time_vector: NDArray[np.float64]
    ) -> list[TemplateResult]:
        """Fit templates to multiple waveforms.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)
        time_vector : ndarray
            Time points in seconds

        Returns
        -------
        list of TemplateResult
            Decomposition results for each subject
        """
        return [self.fit(waveform, time_vector) for waveform in waveforms]
