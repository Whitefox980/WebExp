from planet_nine_toolkit import PlanetNine_AnalysisToolkit

toolkit = PlanetNine_AnalysisToolkit()
result = toolkit.analyze_tno_region("Predicted_Region_A")
toolkit.create_kuiper_belt_treasure_map("outputs/Predicted_Region_A_tno_candidates.csv", "Predicted_Region_A")
