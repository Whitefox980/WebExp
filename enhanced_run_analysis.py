from planet_nine_toolkit import PlanetNine_AnalysisToolkit
import os

toolkit = PlanetNine_AnalysisToolkit()
os.makedirs("outputs", exist_ok=True)

result = toolkit.analyze_tno_region("Predicted_Region_A")
toolkit.create_kuiper_belt_treasure_map("", "Predicted_Region_A")
print("Analysis complete!")
