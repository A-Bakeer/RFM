# Online Retail customer analytics

Cleaning, EDA, RFM and clustering on the Online Retail II data, with a three-page Streamlit dashboard.

## Run it

```bash
pip install -r requirements.txt
python src/build_data.py "path/to/online_retail_II.xlsx"
streamlit run app/app.py
```

The first command only needs the workbook once. It takes about 90 seconds and writes parquet tables to `data/processed`. The dashboard then starts in seconds. To rebuild the notebooks, run `python notebooks/build_notebooks.py`.

## What is here

| Folder | Contents |
|---|---|
| `src/` | `cleaning.py` (cleaning and return matching), `analytics.py` (RFM, clustering, products, cohorts), `build_data.py` (runs it all) |
| `notebooks/` | `01_cleaning_eda.ipynb`, `02_rfm_clustering.ipynb`, and the script that builds them |
| `app/` | The dashboard: Overview, Products, Customers and RFM |
| `data/` | The source data is not copied here. `data/processed` holds generated tables |
| `reports/` | For the final PDF summary |

## Definitions

- **Order:** one invoice. A row in the raw data is a line on an invoice.
- **Net revenue:** quantity times price for goods, less returns matched to an earlier sale by the same customer. Postage, fees and bookkeeping entries are excluded.
- **RFM:** scored as of 10 Dec 2011, on registered customers only. Score 5 is best on all three, so for recency it means fewest days since the last order.
