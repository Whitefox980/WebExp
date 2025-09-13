from planet_nine_toolkit import PlanetNine_AnalysisToolkit
import os

toolkit = PlanetNine_AnalysisToolkit()

# Pronađi preuzeti JWST fajl
jwst_files = []
for root, dirs, files in os.walk("data/mastDownload"):
    for file in files:
        if file.endswith('.fits'):
            jwst_files.append(os.path.join(root, file))

if jwst_files:
    print(f"Koristim JWST fajl: {jwst_files[0]}")
    result = toolkit.analyze_tno_region("Predicted_Region_A", fits_file=jwst_files[0])
else:
    print("Koristim mock podatke")
    result = toolkit.analyze_tno_region("Predicted_Region_A")

toolkit.create_kuiper_belt_treasure_map("outputs/Predicted_Region_A_tno_candidates.csv", "Predicted_Region_A")
