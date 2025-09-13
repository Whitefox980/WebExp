from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PlanetNine_AnalysisToolkit:
    def __init__(self):
        self.tno_properties = {
            'temperature_range': (30, 60),  # K
            'distance_range': (30, 200),  # AU
            'magnitude_range': (20, 23),
            'albedo_range': (0.1, 0.3),
            'spectral_lines': {
                'H2O_ice': [1.5, 2.0, 3.0, 6.0],
                'CH4_ice': [3.3, 7.7],
                'CO2_ice': [4.3, 15.0],
                'NH3': [3.0, 10.0],
                'tholins': [3.4, (5.0, 8.0)],
                'silicates': [(10.0, 12.0)]
            }
        }
        self.planet_nine_theory = {
            'estimated_temperature': 47,
            'predicted_distance': (400, 800),
            'expected_magnitude': 21.7,
            'infrared_excess': 3.2,
            'proper_motion': (0.05, 0.15)
        }
        self.search_regions = [
            {'name': 'Predicted_Region_A', 'ra_center': 3.0, 'dec_center': -20.0, 'search_radius': 15.0},
            {'name': 'Predicted_Region_B', 'ra_center': 15.0, 'dec_center': 10.0, 'search_radius': 10.0}
        ]
        self.webb_filters = {
            'NIRCam': ['F150W2', 'F200W', 'F277W'],
            'MIRI': ['F560W', 'F770W', 'F1000W']
        }
        print("🔭 TNO & Planet Nine Analysis Toolkit Initialized")

    def thermal_signature_modeling(self, temperature, distance_au):
        peak_wavelength = 2897.8 / temperature
        luminosity = 4 * np.pi * (6.371e6 * 2)**2 * 5.67e-8 * temperature**4
        flux = luminosity / (4 * np.pi * (distance_au * 1.496e11)**2)
        return {
            'peak_wavelength_um': peak_wavelength,
            'flux_density': flux,
            'webb_detectable': peak_wavelength < 28,
            'recommended_filter': 'F770W' if 5.6 <= peak_wavelength <= 10 else 'F200W'
        }

    def spectral_line_analysis(self, data, candidate):
        detected_lines = []
        for molecule, wavelengths in self.tno_properties['spectral_lines'].items():
            if isinstance(wavelengths, list):
                for wl in wavelengths:
                    if any(abs(data['wavelength'] - wl) < 0.1 for data_wl in data['spectrum']):
                        detected_lines.append({'molecule': molecule, 'wavelength': wl})
            else:
                wl_range = wavelengths
                if any(wl_range[0] <= data['wavelength'] <= wl_range[1] for data_wl in data['spectrum']):
                    detected_lines.append({'molecule': molecule, 'wavelength': f"{wl_range[0]}-{wl_range[1]}"})
        candidate['spectral_lines'] = detected_lines
        return candidate

    def wise_cross_match(self, candidates, wise_catalog="data/wise_allsky.csv"):
        print("🔍 Cross-matching with WISE/NEOWISE...")
        wise_matches = []
        try:
            wise_data = pd.read_csv(wise_catalog)
            for candidate in candidates:
                ra, dec = candidate['ra_hours'], candidate['dec_degrees']
                matches = wise_data[
                    (abs(wise_data['ra'] - ra) < 0.01) &
                    (abs(wise_data['dec'] - dec) < 0.01)
                ]
                if not matches.empty:
                    for _, match in matches.iterrows():
                        wise_match = {
                            'candidate_id': candidate['candidate_id'],
                            'wise_flux_w1': match['w1mpro'],  # 3.4 μm
                            'wise_flux_w2': match['w2mpro'],  # 4.6 μm
                            'spectral_lines': candidate['spectral_lines']
                        }
                        wise_matches.append(wise_match)
        except FileNotFoundError:
            print("⚠️ WISE catalog missing! Simulating matches...")
            wise_matches = [
                {
                    'candidate_id': c['candidate_id'],
                    'wise_flux_w1': np.random.uniform(10, 15),
                    'wise_flux_w2': np.random.uniform(8, 12),
                    'spectral_lines': c['spectral_lines']
                } for c in candidates
            ]
        return wise_matches

    def analyze_tno_region(self, region_name, fits_file=None):
        print(f"\n🔍 Analyzing TNOs in {region_name}...")
        region = next((r for r in self.search_regions if r['name'] == region_name), None)
        if not region:
            return {"error": "Region not defined"}

        candidates = []
        try:
            if fits_file:
                data = fits.open(fits_file)
                for i in range(5):
                    distance_au = np.random.uniform(30, 200)
                    temperature = np.random.uniform(30, 60)
                    thermal = self.thermal_signature_modeling(temperature, distance_au)
                    candidate = {
                        'candidate_id': f"{region_name}_TNO_{i+1}",
                        'ra_hours': region['ra_center'] + np.random.uniform(-1, 1),
                        'dec_degrees': region['dec_center'] + np.random.uniform(-2, 2),
                        'estimated_magnitude': np.random.uniform(20, 23),
                        'distance_au': distance_au,
                        'thermal_properties': thermal,
                        'analysis_status': 'requires_followup_observation'
                    }
                    mock_spectrum = {'wavelength': [1.5, 3.0, 3.3, 6.0, 7.7], 'spectrum': [0.8, 0.7, 0.9, 0.85, 0.9]}
                    candidate = self.spectral_line_analysis(mock_spectrum, candidate)
                    candidates.append(candidate)
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print("⚠️ FITS file missing! Using mock TNO data...")
            candidates = [
                {
                    'candidate_id': f"{region_name}_TNO_{i+1}",
                    'ra_hours': region['ra_center'] + np.random.uniform(-1, 1),
                    'dec_degrees': region['dec_center'] + np.random.uniform(-2, 2),
                    'estimated_magnitude': np.random.uniform(20, 23),
                    'distance_au': np.random.uniform(30, 200),
                    'thermal_properties': self.thermal_signature_modeling(np.random.uniform(30, 60), np.random.uniform(30, 200)),
                    'spectral_lines': [
                        {'molecule': 'H2O_ice', 'wavelength': 3.0},
                        {'molecule': 'CH4_ice', 'wavelength': 7.7},
                        {'molecule': 'tholins', 'wavelength': '5.0-8.0'}
                    ]
                } for i in range(5)
            ]

        # Cross-match sa WISE/NEOWISE
        wise_matches = self.wise_cross_match(candidates)
        df_wise = pd.DataFrame(wise_matches)
        df_wise.to_csv(f"outputs/{region_name}_wise_matches.csv")

        df = pd.DataFrame(candidates)
        df.to_csv(f"outputs/{region_name}_tno_candidates.csv")
        print(f"📋 Saved {len(candidates)} TNO candidates to outputs/{region_name}_tno_candidates.csv")
        print(f"📋 Saved {len(wise_matches)} WISE matches to outputs/{region_name}_wise_matches.csv")
        return {'region': region, 'candidates': candidates, 'wise_matches': wise_matches}

    def create_kuiper_belt_treasure_map(self, csv_file, region_name, output_file="plots/kuiper_belt_map.png"):
        print(f"🗺️ Generating Kuiper Belt Treasure Map for {region_name}...")
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"⚠️ CSV file {csv_file} not found! Generating mock data...")
            df = pd.DataFrame({
                'candidate_id': [f"{region_name}_TNO_{i+1}" for i in range(10)],
                'ra_hours': np.random.uniform(2, 4, 10),
                'dec_degrees': np.random.uniform(-25, -15, 10),
                'estimated_magnitude': np.random.uniform(20, 23, 10),
                'distance_au': np.random.uniform(30, 200, 10),
                'spectral_lines': [
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}, {'molecule': 'CH4_ice', 'wavelength': 7.7}]),
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}, {'molecule': 'tholins', 'wavelength': '5.0-8.0'}]),
                    str([{'molecule': 'CH4_ice', 'wavelength': 7.7}, {'molecule': 'CO2_ice', 'wavelength': 4.3}]),
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}]),
                    str([{'molecule': 'CH4_ice', 'wavelength': 7.7}, {'molecule': 'NH3', 'wavelength': 10.0}]),
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}, {'molecule': 'tholins', 'wavelength': '5.0-8.0'}]),
                    str([{'molecule': 'CH4_ice', 'wavelength': 7.7}]),
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}, {'molecule': 'CO2_ice', 'wavelength': 4.3}]),
                    str([{'molecule': 'tholins', 'wavelength': '5.0-8.0'}]),
                    str([{'molecule': 'H2O_ice', 'wavelength': 3.0}, {'molecule': 'CH4_ice', 'wavelength': 7.7}, {'molecule': 'NH3', 'wavelength': 10.0}])
                ]
            })

        ra = df['ra_hours']
        dec = df['dec_degrees']
        magnitudes = df['estimated_magnitude']
        spectral_lines = df['spectral_lines'].apply(eval)

        symbol_map = {
            'H2O_ice': {'marker': 'o', 'color': 'cyan', 'label': 'H2O'},
            'CH4_ice': {'marker': '^', 'color': 'blue', 'label': 'CH4'},
            'CO2_ice': {'marker': 's', 'color': 'green', 'label': 'CO2'},
            'NH3': {'marker': 'd', 'color': 'purple', 'label': 'NH3'},
            'tholins': {'marker': '*', 'color': 'red', 'label': 'Tholins'},
            'silicates': {'marker': 'p', 'color': 'orange', 'label': 'Silicates'}
        }

        plt.figure(figsize=(12, 8))
        sns.set_style("darkgrid")
        sns.kdeplot(x=ra, y=dec, cmap="Blues", fill=True, alpha=0.3, label="Candidate Density")

        for i, candidate in df.iterrows():
            spec_lines = eval(candidate['spectral_lines'])
            for line in spec_lines:
                molecule = line['molecule']
                if molecule in symbol_map:
                    plt.scatter(
                        candidate['ra_hours'], candidate['dec_degrees'],
                        s=100 * (24 - candidate['estimated_magnitude']),
                        c=symbol_map[molecule]['color'],
                        marker=symbol_map[molecule]['marker'],
                        alpha=0.7,
                        label=symbol_map[molecule]['label'] if i == 0 else None
                    )

        for i, candidate in df.iterrows():
            spec_lines = eval(candidate['spectral_lines'])
            molecules = [line['molecule'] for line in spec_lines]
            if 'CH4_ice' in molecules and 'tholins' not in molecules:
                plt.scatter(
                    candidate['ra_hours'], candidate['dec_degrees'],
                    s=200, c='yellow', marker='*', edgecolors='black',
                    label='Anomaly (CH4, no tholins)' if i == 0 else None
                )

        plt.xlabel("Right Ascension (hours)")
        plt.ylabel("Declination (degrees)")
        plt.title(f"Kuiper Belt Treasure Map: {region_name}")
        plt.legend()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"✅ Treasure Map saved as {output_file}")

    def generate_observation_proposal(self, candidate):
        ra_h, ra_m, ra_s = int(candidate['ra_hours']), int((candidate['ra_hours'] % 1) * 60), ((candidate['ra_hours'] % 1) * 60 % 1) * 60
        dec_d, dec_m, dec_s = int(abs(candidate['dec_degrees'])), int((abs(candidate['dec_degrees']) % 1) * 60), ((abs(candidate['dec_degrees']) % 1) * 60 % 1) * 60
        dec_sign = '+' if candidate['dec_degrees'] >= 0 else '-'
        return {
            'target_coordinates': f"RA: {ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s, Dec: {dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:05.2f}\"",
            'recommended_instrument': candidate['thermal_properties']['recommended_filter'],
            'estimated_exposure_time': '2-4 hours',
            'spectral_lines': candidate.get('spectral_lines', [])
        }
