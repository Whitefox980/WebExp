from astroquery.mast import Observations
import os

Observations.TIMEOUT = 300
coordinates = "03h00m00s -20d00m00s"
radius = "10deg"

try:
    print("Tražim JWST science observacije...")
    
    obs_table = Observations.query_criteria(
        obs_collection="JWST",
        dataproduct_type="image",
        calib_level=[2, 3]
    )
    
    print(f"Pronađeno {len(obs_table)} JWST science opservacija")
    
    if len(obs_table) > 0:
        # Jednostavno filtriranje bez string operacija
        good_obs = []
        for i, obs in enumerate(obs_table):
            instrument = str(obs['instrument_name'])
            filters = str(obs['filters'])
            
            # Tražimo NIRCam ili MIRI podatke koji nisu kalibracioni
            if ('NIRCAM' in instrument or 'MIRI' in instrument) and 'OPAQUE' not in filters:
                good_obs.append(obs)
                if len(good_obs) >= 3:  # Dovoljno kandidata
                    break
        
        if good_obs:
            target_obs = good_obs[0]
            print(f"Preuzimam: {target_obs['obs_id']}")
            print(f"Instrument: {target_obs['instrument_name']}")
            print(f"Filter: {target_obs['filters']}")
            
            products = Observations.get_product_list(target_obs)
            science_products = Observations.filter_products(products, productType=['SCIENCE'])
            
            if len(science_products) > 0:
                manifest = Observations.download_products(science_products[0:1], download_dir="./data")
                print("Uspešno preuzet science fajl!")
            else:
                print("Nema science proizvoda")
        else:
            print("Nema odgovarajućih imaging opservacija")
    else:
        print("Nema opservacija")
        
except Exception as e:
    print(f"Greška: {e}")
